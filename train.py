import os
import ssl
import urllib3

# 1. SSL verification bypass (must be done before importing ClearML)
urllib3.disable_warnings()
os.environ['CLEARML_API_HOST_VERIFY_CERT'] = '0'
os.environ['CLEARML_FILES_HOST_VERIFY_CERT'] = '0'
os.environ['CLEARML_WEB_HOST_VERIFY_CERT'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

# 2. Patch argparse for Python 3.14 + Hydra compatibility
import argparse
original_expand_help = argparse.HelpFormatter._expand_help
def patched_expand_help(self, action):
    if action.help is not None and not isinstance(action.help, str):
        action.help = str(action.help)
    return original_expand_help(self, action)
argparse.HelpFormatter._expand_help = patched_expand_help

from functools import partial
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from omegaconf import OmegaConf, open_dict

from module import (SIGReg, load_predictor_kernel, load_predictor_gated_residual_kernel,
                    load_predictor_exact_gelu_dropout_kernel,
                    configure_vit_layernorm_exact_gelu_mlp_up,
                    load_vit_layernorm_exact_gelu_mlp_up_kernel)
from utils import get_column_normalizer, get_img_preprocessor, SaveCkptCallback
from clearml import Task


def configure_local_kernels(cfg):
    overrides = []
    for name, kernel_cfg in cfg.get("kernels", {}).items():
        local_path = kernel_cfg.get("local_path")
        repo_id = kernel_cfg.get("repo_id")
        if not local_path or not repo_id:
            continue
        variant = Path(str(local_path))
        metadata = variant / "metadata.json"
        if not metadata.is_file():
            raise FileNotFoundError(f"Local {name} kernel metadata not found: {metadata}")
        overrides.append(f"{repo_id}={variant}")
    if not overrides:
        return
    existing = os.environ.get("LOCAL_KERNELS")
    configured = os.pathsep.join(overrides)
    os.environ["LOCAL_KERNELS"] = (
        f"{existing}{os.pathsep}{configured}" if existing else configured
    )
    print(f"Configured LOCAL_KERNELS={os.environ['LOCAL_KERNELS']}")


def compare_training_gradients(module, eager_loss, fused_loss, tolerances):
    named_params = [(name, p) for name, p in module.model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    eager_grads = torch.autograd.grad(eager_loss, params, retain_graph=True, allow_unused=True)
    delta_grads = torch.autograd.grad(
        eager_loss - fused_loss, params, retain_graph=True, allow_unused=True
    )

    max_abs = 0.0
    diff_sq = torch.zeros((), device=fused_loss.device)
    eager_sq = torch.zeros((), device=fused_loss.device)
    compared = 0
    for (name, _), eager_grad, delta_grad in zip(named_params, eager_grads, delta_grads):
        if eager_grad is None:
            if delta_grad is not None:
                raise RuntimeError(f"SIGReg gradient delta exists for unused parameter {name}")
            continue
        diff = torch.zeros_like(eager_grad, dtype=torch.float32) if delta_grad is None else delta_grad.float()
        max_abs = max(max_abs, diff.abs().max().item())
        diff_sq = diff_sq + diff.square().sum()
        eager_sq = eager_sq + eager_grad.float().square().sum()
        compared += 1

    loss_abs = (eager_loss.detach() - fused_loss.detach()).abs().item()
    grad_rel_l2 = (diff_sq.sqrt() / eager_sq.sqrt().clamp_min(1e-12)).item()
    if loss_abs > tolerances.loss_atol:
        raise RuntimeError(f"SIGReg loss mismatch {loss_abs} exceeds {tolerances.loss_atol}")
    if max_abs > tolerances.grad_atol and grad_rel_l2 > tolerances.grad_rtol:
        raise RuntimeError(
            f"SIGReg gradient mismatch max_abs={max_abs}, rel_l2={grad_rel_l2} "
            f"exceeds atol={tolerances.grad_atol}, rtol={tolerances.grad_rtol}"
        )
    return {
        "sigreg_validation/loss_abs": loss_abs,
        "sigreg_validation/grad_max_abs": max_abs,
        "sigreg_validation/grad_rel_l2": grad_rel_l2,
        "sigreg_validation/parameters_compared": float(compared),
    }

def compare_predictor_kernel(module, eager_pred, fused_pred, eager_loss, fused_loss, predictor, tolerances):
    direct = [((a.float() - b.float()).abs().max()) for a, b in predictor.dual_layernorm_adaln_validation_records()]
    direct_max = torch.stack(direct).max().item() if direct else float("inf")
    downstream_max = (eager_pred.float() - fused_pred.float()).abs().max().item()
    loss_abs = (eager_loss.detach() - fused_loss.detach()).abs().item()
    params = [p for p in predictor.parameters() if p.requires_grad]
    eager_grads = torch.autograd.grad(eager_loss, params, retain_graph=True, allow_unused=True)
    delta_grads = torch.autograd.grad(eager_loss - fused_loss, params, retain_graph=True, allow_unused=True)
    grad_abs = 0.0; diff_sq = torch.zeros((), device=fused_loss.device); ref_sq = torch.zeros_like(diff_sq)
    for ref, delta in zip(eager_grads, delta_grads):
        if ref is None: continue
        delta = torch.zeros_like(ref) if delta is None else delta
        grad_abs = max(grad_abs, delta.float().abs().max().item())
        diff_sq += delta.float().square().sum(); ref_sq += ref.float().square().sum()
    grad_rel = (diff_sq.sqrt() / ref_sq.sqrt().clamp_min(1e-12)).item()
    limits = (tolerances.output_atol, tolerances.downstream_atol, tolerances.loss_atol, tolerances.grad_atol, tolerances.grad_rtol)
    if direct_max > limits[0] or downstream_max > limits[1] or loss_abs > limits[2] or (grad_abs > limits[3] and grad_rel > limits[4]):
        raise RuntimeError(f"Predictor dual-LayerNorm AdaLN mismatch direct={direct_max} downstream={downstream_max} loss={loss_abs} grad_abs={grad_abs} grad_rel={grad_rel}")
    return {"predictor_dual_ln_adaln_validation/direct_max_abs": direct_max,
            "predictor_dual_ln_adaln_validation/downstream_max_abs": downstream_max,
            "predictor_dual_ln_adaln_validation/loss_abs": loss_abs,
            "predictor_dual_ln_adaln_validation/grad_max_abs": grad_abs,
            "predictor_dual_ln_adaln_validation/grad_rel_l2": grad_rel,
            "predictor_dual_ln_adaln_validation/calls": float(len(direct))}

def compare_gated_residual_kernel(module,eager_pred,fused_pred,eager_loss,fused_loss,predictor,tolerances):
    direct=[(a.float()-b.float()).abs().max() for a,b in predictor.gated_residual_gemm_validation_records()]
    direct_max=torch.stack(direct).max().item() if direct else float("inf")
    downstream_max=(eager_pred.float()-fused_pred.float()).abs().max().item();loss_abs=(eager_loss.detach()-fused_loss.detach()).abs().item()
    params=[p for p in predictor.parameters() if p.requires_grad]
    refs=torch.autograd.grad(eager_loss,params,retain_graph=True,allow_unused=True)
    deltas=torch.autograd.grad(eager_loss-fused_loss,params,retain_graph=True,allow_unused=True)
    grad_abs=0.;ds=torch.zeros((),device=fused_loss.device);rs=torch.zeros_like(ds)
    for ref,delta in zip(refs,deltas):
        if ref is None: continue
        delta=torch.zeros_like(ref) if delta is None else delta;grad_abs=max(grad_abs,delta.float().abs().max().item());ds+=delta.float().square().sum();rs+=ref.float().square().sum()
    grad_rel=(ds.sqrt()/rs.sqrt().clamp_min(1e-12)).item()
    if direct_max>tolerances.output_atol or downstream_max>tolerances.downstream_atol or loss_abs>tolerances.loss_atol or (grad_abs>tolerances.grad_atol and grad_rel>tolerances.grad_rtol):
        raise RuntimeError(f"Predictor gated residual GEMM mismatch direct={direct_max} downstream={downstream_max} loss={loss_abs} grad_abs={grad_abs} grad_rel={grad_rel}")
    return {"predictor_gated_residual_gemm_validation/direct_max_abs":direct_max,"predictor_gated_residual_gemm_validation/downstream_max_abs":downstream_max,"predictor_gated_residual_gemm_validation/loss_abs":loss_abs,"predictor_gated_residual_gemm_validation/grad_max_abs":grad_abs,"predictor_gated_residual_gemm_validation/grad_rel_l2":grad_rel,"predictor_gated_residual_gemm_validation/calls":float(len(direct))}

def compare_exact_gelu_dropout_kernel(module,eager_pred,fused_pred,eager_loss,fused_loss,predictor,tolerances):
    direct=[(a.float()-b.float()).abs().max() for a,b in predictor.exact_gelu_dropout_gemm_validation_records()]
    direct_max=torch.stack(direct).max().item() if direct else float("inf")
    downstream_max=(eager_pred.float()-fused_pred.float()).abs().max().item(); loss_abs=(eager_loss.detach()-fused_loss.detach()).abs().item()
    params=[p for p in predictor.parameters() if p.requires_grad]
    refs=torch.autograd.grad(eager_loss,params,retain_graph=True,allow_unused=True); deltas=torch.autograd.grad(eager_loss-fused_loss,params,retain_graph=True,allow_unused=True)
    grad_abs=0.; ds=torch.zeros((),device=fused_loss.device); rs=torch.zeros_like(ds)
    for ref,delta in zip(refs,deltas):
        if ref is None: continue
        delta=torch.zeros_like(ref) if delta is None else delta; grad_abs=max(grad_abs,delta.float().abs().max().item()); ds+=delta.float().square().sum(); rs+=ref.float().square().sum()
    grad_rel=(ds.sqrt()/rs.sqrt().clamp_min(1e-12)).item()
    if direct_max>tolerances.output_atol or downstream_max>tolerances.downstream_atol or loss_abs>tolerances.loss_atol or (grad_abs>tolerances.grad_atol and grad_rel>tolerances.grad_rtol):
        raise RuntimeError(f"Predictor exact-GELU dropout GEMM mismatch direct={direct_max} downstream={downstream_max} loss={loss_abs} grad_abs={grad_abs} grad_rel={grad_rel}")
    return {"predictor_exact_gelu_dropout_gemm_validation/direct_max_abs":direct_max,"predictor_exact_gelu_dropout_gemm_validation/downstream_max_abs":downstream_max,"predictor_exact_gelu_dropout_gemm_validation/loss_abs":loss_abs,"predictor_exact_gelu_dropout_gemm_validation/grad_max_abs":grad_abs,"predictor_exact_gelu_dropout_gemm_validation/grad_rel_l2":grad_rel,"predictor_exact_gelu_dropout_gemm_validation/calls":float(len(direct))}

def compare_vit_mlp_up_kernel(module,layers,eager_emb,fused_emb,eager_loss,fused_loss,tolerances):
    diffs=[d for layer in layers for d in layer._lewm_mlp_up_records]
    direct_max=torch.stack(diffs).max().item() if diffs else float("inf")
    downstream_max=(eager_emb.float()-fused_emb.float()).abs().max().item();loss_abs=(eager_loss.detach()-fused_loss.detach()).abs().item()
    layer=layers[0];x=layer._lewm_mlp_up_sample.detach().requires_grad_(True);ln=layer.layernorm_after
    fc1=layer.mlp.fc1 if layer._lewm_mlp_up_layout=="modern" else layer.intermediate.dense
    act=layer.mlp.activation_fn if layer._lewm_mlp_up_layout=="modern" else layer.intermediate.intermediate_act_fn
    eager=act(fc1(ln(x)));fused=load_vit_layernorm_exact_gelu_mlp_up_kernel().vit_layernorm_exact_gelu_mlp_up(x,ln.weight,ln.bias,fc1.weight,fc1.bias,ln.eps)
    params=[x,ln.weight,ln.bias,fc1.weight,fc1.bias]
    refs=torch.autograd.grad(eager.float().square().mean(),params,retain_graph=True,allow_unused=True);deltas=torch.autograd.grad(eager.float().square().mean()-fused.float().square().mean(),params,retain_graph=True,allow_unused=True)
    grad_abs=0.;ds=torch.zeros((),device=fused_loss.device);rs=torch.zeros_like(ds)
    for ref,delta in zip(refs,deltas):
        if ref is None: continue
        delta=torch.zeros_like(ref) if delta is None else delta;grad_abs=max(grad_abs,delta.float().abs().max().item());ds+=delta.float().square().sum();rs+=ref.float().square().sum()
    grad_rel=(ds.sqrt()/rs.sqrt().clamp_min(1e-12)).item()
    if direct_max>tolerances.output_atol or downstream_max>tolerances.downstream_atol or loss_abs>tolerances.loss_atol or (grad_abs>tolerances.grad_atol and grad_rel>tolerances.grad_rtol):
        raise RuntimeError(f"ViT MLP-up mismatch direct={direct_max} downstream={downstream_max} loss={loss_abs} grad_abs={grad_abs} grad_rel={grad_rel}")
    return {"vit_ln_mlp_up_gelu_validation/direct_max_abs":direct_max,"vit_ln_mlp_up_gelu_validation/downstream_max_abs":downstream_max,"vit_ln_mlp_up_gelu_validation/loss_abs":loss_abs,"vit_ln_mlp_up_gelu_validation/grad_max_abs":grad_abs,"vit_ln_mlp_up_gelu_validation/grad_rel_l2":grad_rel,"vit_ln_mlp_up_gelu_validation/calls":float(len(diffs))}

def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses."""

    ctx_len = cfg.history_size
    n_preds = cfg.num_preds
    lambd = cfg.loss.sigreg.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)

    base_model=getattr(self.model,"_orig_mod",self.model);vit_layers=base_model._lewm_vit_mlp_up_layers
    vit_mode=str(cfg.model.encoder.get("mlp_up_implementation","eager"));eager_encoder_output=None
    if self.training and vit_mode=="validate":
        cpu_rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state(batch["pixels"].device)
        for layer in vit_layers: layer._lewm_mlp_up_mode="eager"
        eager_batch={k:(v.clone() if torch.is_tensor(v) else v) for k,v in batch.items()}
        with torch.no_grad(): eager_encoder_output=self.model.encode(eager_batch)
        torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng,batch["pixels"].device)
        for layer in vit_layers: layer._lewm_mlp_up_mode="validate";layer._lewm_mlp_up_records.clear();layer._lewm_mlp_up_sample=None
    output = self.model.encode(batch)

    emb = output["emb"]  # (B, T, D)
    act_emb = output["act_emb"]

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, : ctx_len]

    tgt_emb = emb[:, n_preds:] # label
    base_model = getattr(self.model, "_orig_mod", self.model)
    predictor = base_model.predictor
    predictor_mode = cfg.model.predictor.get("dual_layernorm_adaln_implementation", "eager")
    gated_mode = cfg.model.predictor.get("gated_residual_gemm_implementation", "eager")
    gelu_mode = cfg.model.predictor.get("exact_gelu_dropout_gemm_implementation", "eager")
    eager_pred_emb = None
    if self.training and (predictor_mode == "validate" or gated_mode == "validate" or gelu_mode == "validate"):
        cpu_rng = torch.random.get_rng_state(); cuda_rng = torch.cuda.get_rng_state(ctx_emb.device)
        predictor.set_dual_layernorm_adaln_implementation("eager")
        predictor.set_gated_residual_gemm_implementation("eager")
        predictor.set_exact_gelu_dropout_gemm_implementation("eager")
        eager_pred_emb = self.model.predict(ctx_emb, ctx_act)
        torch.random.set_rng_state(cpu_rng); torch.cuda.set_rng_state(cuda_rng, ctx_emb.device)
        predictor.clear_dual_layernorm_adaln_validation_records()
        predictor.clear_gated_residual_gemm_validation_records()
        predictor.clear_exact_gelu_dropout_gemm_validation_records()
        predictor.set_dual_layernorm_adaln_implementation("validate" if predictor_mode == "validate" else predictor_mode)
        predictor.set_gated_residual_gemm_implementation("validate" if gated_mode == "validate" else gated_mode)
        predictor.set_exact_gelu_dropout_gemm_implementation("validate" if gelu_mode == "validate" else gelu_mode)
    eager_vit_pred=None
    if eager_encoder_output is not None:
        eager_ctx=eager_encoder_output["emb"][:,:ctx_len];eager_act=eager_encoder_output["act_emb"][:,:ctx_len]
        cpu_rng=torch.random.get_rng_state();cuda_rng=torch.cuda.get_rng_state(ctx_emb.device)
        with torch.no_grad(): eager_vit_pred=self.model.predict(eager_ctx,eager_act)
        torch.random.set_rng_state(cpu_rng);torch.cuda.set_rng_state(cuda_rng,ctx_emb.device)
    pred_emb = self.model.predict(ctx_emb, ctx_act) # pred

    # LeWM loss
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    is_training = self.training
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1), validate=is_training)
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  

    if eager_encoder_output is not None:
        eager_emb=eager_encoder_output["emb"];eager_tgt=eager_emb[:,n_preds:]
        # SIGReg is shared and samples random projections; reuse its actual loss
        # so the comparison isolates only the encoder branch without advancing RNG.
        with torch.no_grad(): eager_pred_loss=(eager_vit_pred-eager_tgt).pow(2).mean();eager_total=eager_pred_loss+lambd*output["sigreg_loss"].detach()
        metrics=compare_vit_mlp_up_kernel(self,vit_layers,eager_emb,emb,eager_total,output["loss"],cfg.kernels.vit_layernorm_exact_gelu_mlp_up.validation)
        self.log_dict(metrics,on_step=True,on_epoch=False,sync_dist=False)
        print(f"ViT MLP-up kernel training validation: metrics={metrics}, module={load_vit_layernorm_exact_gelu_mlp_up_kernel().__file__}")

    if eager_pred_emb is not None:
        eager_pred_loss = (eager_pred_emb - tgt_emb).pow(2).mean()
        if gelu_mode == "validate":
            metrics=compare_exact_gelu_dropout_kernel(self,eager_pred_emb,pred_emb,eager_pred_loss,output["pred_loss"],predictor,cfg.kernels.predictor_exact_gelu_dropout_gemm.validation)
            loaded=load_predictor_exact_gelu_dropout_kernel().__file__
        elif gated_mode == "validate":
            metrics=compare_gated_residual_kernel(self,eager_pred_emb,pred_emb,eager_pred_loss,output["pred_loss"],predictor,cfg.kernels.predictor_gated_residual_gemm.validation)
            loaded=load_predictor_gated_residual_kernel().__file__
        else:
            metrics = compare_predictor_kernel(self, eager_pred_emb, pred_emb, eager_pred_loss,
                output["pred_loss"], predictor, cfg.kernels.predictor_dual_layernorm_adaln.validation); loaded=load_predictor_kernel().__file__
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=False)
        print(f"Predictor kernel training validation: metrics={metrics}, module={loaded}")

    if is_training and self.sigreg.comparison_active:
        eager_total = output["pred_loss"] + lambd * self.sigreg.validation_eager_loss
        tolerances = (
            cfg.kernels.projection_normalization.validation
            if self.sigreg.normalization_comparison_active
            else cfg.kernels.sigreg.validation
        )
        metrics = compare_training_gradients(
            self, eager_total, output["loss"], tolerances
        )
        self.log_dict(metrics, on_step=True, on_epoch=False, sync_dist=False)
        print(
            "Kernel training-step validation: "
            f"sigreg_calls={self.sigreg.validation_calls}, "
            f"normalization_calls={self.sigreg.normalization_validation_calls}, "
            f"metrics={metrics}"
        )

    if stage in ["validate", "val"]:
        losses_epoch = {f"validate_epoch/{k}_epoch": v.detach() for k, v in output.items() if "loss" in k}
        self.log_dict(losses_epoch, on_step=False, on_epoch=True, sync_dist=True)
        
        losses_step = {f"validate_step/{k}_step": v.detach() for k, v in output.items() if "loss" in k}
        self.log_dict(losses_step, on_step=True, on_epoch=False, sync_dist=True)
    else:
        losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
        self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output

class BatchedSubset(torch.utils.data.Subset):
    def __getitems__(self, indices):
        underlying_indices = [self.indices[i] for i in indices]
        if hasattr(self.dataset, '__getitems__'):
            return self.dataset.__getitems__(underlying_indices)
        return [self.dataset[i] for i in underlying_indices]

@hydra.main(version_base=None, config_path="./config/train", config_name="lewm")
def run(cfg):
    #########################
    ##       ClearML       ##
    #########################
    # Initialize the ClearML task under hierarchical path LeWM/Training
    task = Task.init(
        project_name="LeWM/Training",
        task_name=f"LeWM-Train-{cfg.output_model_name}",
        task_type=Task.TaskTypes.training,
        auto_connect_frameworks={"pytorch": False, "hydra": False},
        output_uri=True
    )

    # Connect Hydra configs to task parameters
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_dict)
    
    # Merge overrides from server back into cfg
    with open_dict(cfg):
        cfg.merge_with(cfg_dict)

    configure_local_kernels(cfg)

    # Sync and update ClearML's Configuration tab named "OmegaConf" to reflect the actual resolved/merged config
    task.set_configuration_object("OmegaConf", OmegaConf.to_yaml(cfg))

    # Extract dataset options
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")

    # Append new tags while preserving any enqueued preset tags
    existing_tags = task.get_tags() or []
    task.set_tags(list(set(existing_tags + ["base-experiment", f"data:{dataset_name}"])))

    clearml_id = dataset_cfg.pop("clearml_id", None)
    clearml_name = dataset_cfg.pop("clearml_name", None)
    clearml_project = dataset_cfg.pop("clearml_project", "LeWM")

    downloaded_path = None
    if clearml_id or clearml_name:
        from clearml import Dataset
        if clearml_id:
            print(f"📥 Fetching ClearML dataset by ID: {clearml_id}...")
            clearml_ds = Dataset.get(dataset_id=clearml_id)
        else:
            print(f"📥 Fetching ClearML dataset by Name: {clearml_name} in project {clearml_project}...")
            clearml_ds = Dataset.get(dataset_name=clearml_name, dataset_project=clearml_project)
        
        downloaded_path = clearml_ds.get_local_copy()
        print(f"📂 Dataset downloaded to local copy at: {downloaded_path}")
        cache_dir = downloaded_path
        dataset_name = os.path.join(downloaded_path, dataset_name)
    else:
        cache_dir = os.environ.get("LOCAL_DATASET_DIR", None)

    try:
        #########################
        ##       dataset       ##
        #########################
        dataset = swm.data.load_dataset(
            dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
        )
        transforms = [get_img_preprocessor(source='pixels', target='pixels', img_size=cfg.img_size)]
        
        with open_dict(cfg):
            for col in cfg.data.dataset.keys_to_load:
                if col.startswith("pixels"):
                    continue
                normalizer = get_column_normalizer(dataset, col, col)
                transforms.append(normalizer)

            cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

        transform = spt.data.transforms.Compose(*transforms)
        dataset.transform = transform

        rnd_gen = torch.Generator().manual_seed(cfg.seed)
        train_set, val_set = spt.data.random_split(
            dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=rnd_gen
        )
        train_set = BatchedSubset(train_set.dataset, train_set.indices)
        val_set = BatchedSubset(val_set.dataset, val_set.indices)

        # Configure loaders to prevent memory bloat (especially during validation sanity/full check)
        train_loader_cfg = dict(cfg.loader)
        
        val_loader_cfg = dict(cfg.loader)
        val_loader_cfg["num_workers"] = min(2, cfg.loader.num_workers)
        val_loader_cfg["persistent_workers"] = False

        train = torch.utils.data.DataLoader(train_set, **train_loader_cfg, shuffle=True, drop_last=True, generator=rnd_gen)
        val = torch.utils.data.DataLoader(val_set, **val_loader_cfg, shuffle=False, drop_last=False)
        
        # Calculate stepping parameters to avoid PyTorch Lightning's CombinedLoader bug with Lance datasets
        steps_per_epoch = len(train)
        scheduler_max_epochs = cfg.get("scheduler_max_epochs", cfg.trainer.max_epochs)
        max_steps = steps_per_epoch * scheduler_max_epochs
        warmup_steps = int(0.01 * max_steps)
        
        print(f"📊 Manually calculated scheduler steps: max_steps={max_steps}, warmup_steps={warmup_steps} (based on {scheduler_max_epochs} scheduler epochs)")
        
        ##############################
        ##       model / optim      ##
        ##############################

        encoder_mode=str(cfg.model.encoder.get("mlp_up_implementation", "eager"))
        encoder_kernel_repo=str(cfg.model.encoder.get("mlp_up_kernel_repo_id", cfg.kernels.vit_layernorm_exact_gelu_mlp_up.repo_id))
        model_cfg=OmegaConf.create(OmegaConf.to_container(cfg.model,resolve=True))
        model_cfg.encoder.pop("mlp_up_implementation",None);model_cfg.encoder.pop("mlp_up_kernel_repo_id",None)
        world_model = hydra.utils.instantiate(model_cfg)
        world_model._lewm_vit_mlp_up_layers=configure_vit_layernorm_exact_gelu_mlp_up(
            world_model.encoder,encoder_mode,encoder_kernel_repo)
        if cfg.get("compile", True):
            print("⚡ Compiling base model with torch.compile...")
            world_model = torch.compile(world_model)

        optimizers = {
            'model_opt': {
                "modules": 'model',
                "optimizer": dict(cfg.optimizer),
                "scheduler": {
                    "type": cfg.get("scheduler_type", "LinearWarmupCosineAnnealingLR"),
                    "warmup_steps": warmup_steps,
                    "max_steps": max_steps,
                    "warmup_start_lr": 0.0,
                    "eta_min": 0.0,
                },
                "interval": cfg.get("scheduler_interval", "epoch"),
            },
        }

        data_module = spt.data.DataModule(train=train, val=val)
        world_model = spt.Module(
            model = world_model,
            sigreg = SIGReg(**cfg.loss.sigreg.kwargs),
            forward=partial(lejepa_forward, cfg=cfg),
            optim=optimizers,
        )

        ##########################
        ##       training       ##
        ##########################

        run_id = cfg.get("subdir") or ""
        run_dir = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id)

        loggers = []
        tb_logger = TensorBoardLogger(save_dir=str(run_dir), name="tb_logs")
        loggers.append(tb_logger)

        if cfg.wandb.enabled:
            wandb_logger = WandbLogger(**cfg.wandb.config)
            wandb_logger.log_hyperparams(OmegaConf.to_container(cfg))
            loggers.append(wandb_logger)

        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            OmegaConf.save(cfg, f)

        # Initialize callback with max epochs to enable publishing the final model checkpoint
        object_dump_callback = SaveCkptCallback(
            run_name=cfg.output_model_name,
            cfg=cfg.model,
            epoch_interval=cfg.get("epoch_interval", 1),
            max_epochs=cfg.trainer.max_epochs
        )

        trainer = pl.Trainer(
            **cfg.trainer,
            callbacks=[object_dump_callback],
            num_sanity_val_steps=1,
            logger=loggers,
            enable_checkpointing=True,
        )

        ckpt_path = run_dir / f"{cfg.output_model_name}_weights.ckpt"
        manager = spt.Manager(
            trainer=trainer,
            module=world_model,
            data=data_module,
            ckpt_path=ckpt_path if ckpt_path.exists() else None,
        )

        manager()
    finally:
        # Guarantee local dataset cleanup to conserve disk space
        if downloaded_path:
            import shutil
            print(f"🧹 Cleaning up local dataset copy at {downloaded_path}...")
            shutil.rmtree(downloaded_path, ignore_errors=True)

if __name__ == "__main__":
    run()
