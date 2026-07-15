import os

os.environ["MUJOCO_GL"] = "egl"

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

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
from clearml import Task, Logger

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

def load_pretrained_compat(name: str):
    from hydra.utils import instantiate
    import torch

    cache_dir = swm.data.utils.get_cache_dir(None, sub_folder='checkpoints')
    
    # Check if this is a ClearML model ID or URI
    is_clearml_id = name.startswith("model:") or (len(name) == 32 and all(c in "0123456789abcdef" for c in name.lower()))
    
    if is_clearml_id:
        model_id = name.replace("model:", "", 1)
        from clearml import Model
        from omegaconf import OmegaConf
        print(f"📥 Fetching ClearML model weights for ID: {model_id}...")
        clearml_model = Model(model_id=model_id)
        checkpoint_path = Path(clearml_model.get_local_copy())

        # Load configuration from model config_text if stored (OmegaConf/Hydra format string)
        config_text = clearml_model.config_text or ""
        if config_text.strip():
            print("📋 Loading config from model config_text...")
            config = OmegaConf.to_container(OmegaConf.create(config_text), resolve=True)
        else:
            # Fallback to local config.json
            config_path = checkpoint_path.parent / 'config.json'
            if not config_path.exists():
                config_path = cache_dir / 'lewm' / 'config.json'
            if not config_path.exists():
                raise FileNotFoundError(f'Could not find config.json for ClearML model {model_id}')
            import json
            with open(config_path) as f:
                config = json.load(f)
    elif name.endswith('.ckpt'):
        import json
        local_path = cache_dir / name
        if not local_path.exists():
            if Path(name).exists():
                local_path = Path(name)
            else:
                raise FileNotFoundError(f'Checkpoint not found: {name}')
        checkpoint_path = local_path
        
        # Load config.json
        config_path = checkpoint_path.parent / 'config.json'
        if not config_path.exists():
            config_path = checkpoint_path.parent.parent / 'config.json'
        if not config_path.exists():
            config_path = cache_dir / 'lewm' / 'config.json'
            
        if not config_path.exists():
            raise FileNotFoundError(f'Could not find config.json for {name}')
            
        with open(config_path) as f:
            config = json.load(f)
    else:
        checkpoint_path, config = swm.wm.utils._resolve(name, cache_dir)
        
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
        # Strip PyTorch Lightning 'model.' wrapper prefix if present
        state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()}

    # Clean state dict keys from torch.compile's '_orig_mod.' prefix
    state_dict = {k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k: v for k, v in state_dict.items()}

    model = instantiate(config)

    model_state = model.state_dict()
    if not all(k in state_dict for k in model_state.keys()):
        print("🔄 Adapting checkpoint state dict keys to match current transformers library version...")
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if 'encoder.encoder.layer.' in k:
                new_key = k.replace('encoder.encoder.layer.', 'encoder.layers.')
                new_key = new_key.replace('.attention.attention.query.', '.attention.q_proj.')
                new_key = new_key.replace('.attention.attention.key.', '.attention.k_proj.')
                new_key = new_key.replace('.attention.attention.value.', '.attention.v_proj.')
                new_key = new_key.replace('.attention.output.dense.', '.attention.o_proj.')
                new_key = new_key.replace('.intermediate.dense.', '.mlp.fc1.')
                new_key = new_key.replace('.output.dense.', '.mlp.fc2.')
            new_state_dict[new_key] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    return model

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    #########################
    ##       ClearML       ##
    #########################
    # Initialize ClearML under LeWM/Evaluation hierarchy
    task = Task.init(
        project_name="LeWM/Evaluation",
        task_name=f"LeWM-Eval-{cfg.eval.dataset_name}",
        task_type=Task.TaskTypes.testing,
        auto_connect_frameworks={"pytorch": False, "hydra": False}
    )

    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    task.connect(cfg_container)
    with open_dict(cfg):
        cfg.policy = cfg_container.get("policy", cfg.policy)

    # Sync and update ClearML's Configuration tab named "OmegaConf" to reflect the actual resolved/merged config
    task.set_configuration_object("OmegaConf", OmegaConf.to_yaml(cfg))

    if not task.get_tags():
        task.set_tags(["base-experiment", "evaluation", f"data:{cfg.eval.dataset_name}"])

    # Resolve ClearML dataset if specified
    clearml_id = cfg.eval.get("clearml_id", None)
    clearml_name = cfg.eval.get("clearml_name", None)
    clearml_project = cfg.eval.get("clearml_project", "LeWM")

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
        with open_dict(cfg):
            cfg.cache_dir = downloaded_path
            cfg.eval.dataset_name = os.path.join(downloaded_path, cfg.eval.dataset_name)

    try:
        # create world environment
        cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
        world = swm.World(**cfg.world, image_shape=(224, 224))

        # create the transform
        transform = {
            "pixels": img_transform(cfg),
            "goal": img_transform(cfg),
        }

        dataset = get_dataset(cfg, cfg.eval.dataset_name)
        stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

        process = {}
        for col in cfg.dataset.keys_to_cache:
            if col in ["pixels"]:
                continue
            processor = preprocessing.StandardScaler()
            col_data = stats_dataset.get_col_data(col)
            col_data = col_data[~np.isnan(col_data).any(axis=1)]
            processor.fit(col_data)
            process[col] = processor

            if col != "action":
                process[f"goal_{col}"] = process[col]

        # -- run evaluation
        policy = cfg.get("policy", "random")
        is_clearml_id = policy.startswith("model:") or (len(policy) == 32 and all(c in "0123456789abcdef" for c in policy.lower()))

        if policy != "random":
            model = load_pretrained_compat(cfg.policy)
            model = model.to("cuda")
            model = model.eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True
            config = swm.PlanConfig(**cfg.plan_config)
            solver = hydra.utils.instantiate(cfg.solver, model=model)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process=process, transform=transform
            )

        else:
            policy = swm.policy.RandomPolicy()

        results_path = (
            Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
            if cfg.policy != "random" and not is_clearml_id
            else Path(__file__).parent
        )

        # sample the episodes and the starting indices
        episode_len = get_episodes_length(dataset, ep_indices)
        max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
        max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
        # Map each dataset row’s episode_idx to its max_start_idx
        col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
        max_start_per_row = np.array(
            [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
        )

        # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
        valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
        valid_indices = np.nonzero(valid_mask)[0]
        print(valid_mask.sum(), "valid starting points found for evaluation.")

        g = np.random.default_rng(cfg.seed)
        random_episode_indices = g.choice(
            len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
        )

        # sort increasingly to avoid issues with HDF5Dataset indexing
        random_episode_indices = np.sort(valid_indices[random_episode_indices])

        print(random_episode_indices)

        eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
        eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

        if len(eval_episodes) < cfg.eval.num_eval:
            raise ValueError("Not enough episodes with sufficient length for evaluation.")

        world.set_policy(policy)

        results_path.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video=results_path,
        )
        end_time = time.time()
        
        print(metrics)

        # Report metrics to ClearML
        clearml_logger = Logger.current_logger()
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float, np.number)):
                    clearml_logger.report_scalar(title="Evaluation Metrics", series=k, value=float(v), iteration=0)

        results_path = results_path / cfg.output.filename
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with results_path.open("a") as f:
            f.write("\n")  # separate from previous runs

            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")

            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {metrics}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")

    finally:
        # Guarantee dataset cleanup on local agent
        if downloaded_path:
            import shutil
            print(f"🧹 Cleaning up local dataset copy at {downloaded_path}...")
            shutil.rmtree(downloaded_path, ignore_errors=True)


if __name__ == "__main__":
    run()
