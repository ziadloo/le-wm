import argparse
import os
import ssl
import urllib3
import sys

# SSL bypass flags (must be set before importing ClearML)
urllib3.disable_warnings()
os.environ['CLEARML_API_HOST_VERIFY_CERT'] = '0'
os.environ['CLEARML_FILES_HOST_VERIFY_CERT'] = '0'
os.environ['CLEARML_WEB_HOST_VERIFY_CERT'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

from omegaconf import OmegaConf
from clearml import Task, Model

DEFAULT_TRAIN_PACKAGES = [
    "datasets>=3.0.0",
    "pyarrow>=20.0.0",
    "pygame-ce",
    "pymunk",
    "shapely",
    "hdf5plugin",
    "tensorboard",
    "tensorboardX"
]

DEFAULT_EVAL_PACKAGES = DEFAULT_TRAIN_PACKAGES + ["imageio-ffmpeg"]

# Presets map to decouple target task details from conditional structures
PRESETS = {
    "baseline_tworoom": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-tworoom-sigreg-bf16_true",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "tworoom")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/tworoom.yaml",
        "dataset_name": "tworoom.lance",
        "clearml_dataset_name": "LeWM-TwoRoom",
        "overrides": {
            "trainer.max_epochs": 20,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-true",
            "loader.batch_size": 144,
            "num_workers": 4,
            "loader.prefetch_factor": 2,
            "compile": True,
        },
        "tags": ["baseline", "enqueued", "tworoom", "lance-format", "bf16-true", "data:tworoom.lance"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "baseline_reacher": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-reacher-sigreg-bf16_true",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "dmc")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/dmc.yaml",
        "dataset_name": "reacher.lance",
        "clearml_dataset_name": "LeWM-Reacher",
        "overrides": {
            "trainer.max_epochs": 20,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-true",
            "loader.batch_size": 144,
            "num_workers": 4,
            "loader.prefetch_factor": 2,
            "compile": True,
        },
        "tags": ["baseline", "enqueued", "reacher", "lance-format", "bf16-true", "data:reacher.lance"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "baseline_cube": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-cube_single-sigreg-bf16_true",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "ogb")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/ogb.yaml",
        "dataset_name": "ogbench/cube_single_expert.lance",
        "clearml_dataset_name": "LeWM-Cube",
        "overrides": {
            "trainer.max_epochs": 20,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-true",
            "loader.batch_size": 144,
            "num_workers": 4,
            "loader.prefetch_factor": 2,
            "compile": True,
        },
        "tags": ["baseline", "enqueued", "cube_single", "lance-format", "bf16-true", "data:ogbench/cube_single_expert.lance"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "pusht_lance": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-pusht_lance-sigreg-bf16_mixed",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "pusht_lance")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/pusht_lance.yaml",
        "dataset_name": "pusht_expert_train.lance",
        "clearml_dataset_name": "LeWM-PushT",
        "overrides": {
            "trainer.max_epochs": 20,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16",
            "loader.batch_size": 144,
            "loader.prefetch_factor": 2,
            "num_workers": 4,
            "compile": True,
        },
        "tags": ["base-experiment", "bf16-mixed", "lance", "pusht", "sigreg"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "pusht_amp": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-pusht-sigreg-bf16_true",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "pusht")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/pusht.yaml",
        "dataset_name": "pusht_expert_train.lance",
        "clearml_dataset_name": "LeWM-PushT",
        "overrides": {
            "trainer.max_epochs": 20,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-true",
            "loader.batch_size": 144,
            "loader.prefetch_factor": 2,
            "num_workers": 4,
            "compile": True,
        },
        "tags": ["base-experiment", "bf16-true", "lance", "pusht", "sigreg"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "pusht_vicreg": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-pusht_lance-vicreg-bf16_mixed",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "pusht_lance")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/pusht_lance.yaml",
        "dataset_name": "pusht_expert_train.lance",
        "clearml_dataset_name": "LeWM-PushT",
        "overrides": {
            "trainer.max_epochs": 20,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-mixed",
            "loader.batch_size": 144,
            "loader.prefetch_factor": 2,
            "num_workers": 4,
            "compile": True,
        },
        "tags": ["enqueued", "agent-worker", "lance-format", "vicreg", "bf16-mixed", "data:pusht_expert_train.lance"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "pusht_visreg": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-pusht_lance-visreg_w5.0-bf16_mixed",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "pusht_lance")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/pusht_lance.yaml",
        "dataset_name": "pusht_expert_train.lance",
        "clearml_dataset_name": "LeWM-PushT",
        "overrides": {
            "trainer.max_epochs": 20,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16",
            "loader.batch_size": 144,
            "loader.prefetch_factor": 2,
            "num_workers": 4,
            "compile": True,
            "loss.visreg.weight": 5.0,
        },
        "tags": ["enqueued", "agent-worker", "lance-format", "visreg_w5.0", "bf16-mixed", "data:pusht_expert_train.lance"],
        "packages": DEFAULT_TRAIN_PACKAGES
    }
}

def load_and_merge_configs(config_path, data_config_path=None, overrides=None):
    """Loads Hydra/OmegaConf configurations and applies custom dictionaries overrides."""
    cfg = OmegaConf.load(config_path)
    if data_config_path:
        data_cfg = OmegaConf.load(data_config_path)
        cfg = OmegaConf.merge(cfg, data_cfg)
    
    cfg_dict = OmegaConf.to_container(cfg, resolve=False)
    
    # Apply nested key overrides
    if overrides:
        for k, v in overrides.items():
            keys = k.split('.')
            d = cfg_dict
            for key in keys[:-1]:
                d = d.setdefault(key, {})
            d[keys[-1]] = v
            
    return cfg_dict

def enqueue_task(task_spec, queue_name="default"):
    """Core function to create and enqueue tasks programmatically."""
    print(f"🚀 Creating task '{task_spec['task_name']}' in project '{task_spec['project_name']}'...")
    
    # Load and merge configurations
    cfg_dict = load_and_merge_configs(
        config_path=task_spec["config_path"],
        data_config_path=task_spec.get("data_config_path"),
        overrides=task_spec.get("overrides")
    )
    
    # Force dynamic dataset binding values inside OmegaConf
    if "data" in cfg_dict and "dataset" in cfg_dict["data"]:
        cfg_dict["data"]["dataset"]["name"] = task_spec.get("dataset_name")
        cfg_dict["data"]["dataset"]["clearml_name"] = task_spec.get("clearml_dataset_name")
        cfg_dict["data"]["dataset"]["clearml_project"] = "LeWM"
        
    task = Task.create(
        project_name=task_spec["project_name"],
        task_name=task_spec["task_name"],
        task_type=task_spec["task_type"],
        script=task_spec["script"],
        argparse_args=task_spec.get("argparse_args", []),
        force_single_script_file=False,
        detect_repository=True,
        packages=task_spec.get("packages", DEFAULT_TRAIN_PACKAGES)
    )
    
    task.connect(cfg_dict)
    task.set_tags(task_spec.get("tags", []))
    
    print(f"📦 Enqueueing Task ID {task.id} to queue '{queue_name}'...")
    Task.enqueue(task=task, queue_name=queue_name)
    print(f"✅ Successfully enqueued task ID: {task.id}")
    print(f"🔗 ClearML Task Link: {task.get_output_log_web_page()}\n")
    return task.id

def handle_preset_mode(preset_name, queue_name):
    """Handles execution of one or multiple preset tasks using a dictionary lookup."""
    if preset_name == "all_baselines":
        # Launch baseline training targets
        targets = ["baseline_tworoom", "baseline_reacher", "baseline_cube"]
        for target in targets:
            enqueue_task(PRESETS[target], queue_name)
    elif preset_name in PRESETS:
        enqueue_task(PRESETS[preset_name], queue_name)
    else:
        print(f"❌ Error: Preset '{preset_name}' not found. Available presets: {list(PRESETS.keys())} or 'all_baselines'.")
        sys.exit(1)

def handle_eval_checkpoint(checkpoint, queue_name):
    """Creates and enqueues evaluation tasks for a specific checkpoint or all epochs."""
    # Ensure correct relative pathing
    if checkpoint.startswith("checkpoints/"):
        checkpoint = checkpoint[len("checkpoints/"):]
        
    epoch_label = os.path.splitext(os.path.basename(checkpoint))[0]
    
    task_spec = {
        "project_name": "LeWM/Evaluation",
        "task_name": f"LeWM-Eval-pusht_expert_train-{epoch_label}",
        "task_type": Task.TaskTypes.testing,
        "script": "eval.py",
        "argparse_args": [("policy", checkpoint)],
        "config_path": "config/eval/pusht.yaml",
        "overrides": {
            "policy": checkpoint,
            "eval.dataset_name": "pusht_expert_train.lance",
            "dataset.stats": "pusht_expert_train.lance",
            "eval.clearml_name": "LeWM-PushT",
            "eval.clearml_project": "LeWM"
        },
        "tags": ["evaluation", f"checkpoint:{epoch_label}", "data:pusht_expert_train", "model:lewm"],
        "packages": DEFAULT_EVAL_PACKAGES
    }
    
    enqueue_task(task_spec, queue_name)

def handle_evaluate_job(job_id, queue_name):
    """Fetches checkpoints registered by a training job and schedules evaluation tasks."""
    print(f"🔍 Fetching training task ID: {job_id}...")
    t = Task.get_task(task_id=job_id)
    
    # Find dataset name parameter
    try:
        training_dataset_name = t.data.configuration.get('OmegaConf').value.split("name: ")[1].split("\n")[0].strip()
    except Exception:
        training_dataset_name = "pusht_expert_train.lance"
        
    models = t.get_models().get('output', [])
    task_models = [m for m in models if m.task == job_id]
    
    if not task_models:
        print(f"❌ Error: No output models found registered by task {job_id}.")
        return

    print(f"📊 Found {len(task_models)} output models. Scheduling evaluation jobs...")
    
    # Process each registered model
    for m in task_models:
        # Determine epoch from tags
        epoch = None
        for tag in m.tags:
            if tag.startswith("epoch:"):
                try:
                    epoch = int(tag.split(":")[1])
                except ValueError:
                    pass
                break
                
        if epoch is None:
            # Fallback to name parsing if tags are missing
            try:
                epoch = int(m.name.split("-epoch-")[1])
            except Exception:
                continue
                
        epoch_str = f"{epoch:03d}"
        task_spec = {
            "project_name": "LeWM/Evaluation",
            "task_name": f"LeWM-Eval-{epoch_str}-{t.name}",
            "task_type": Task.TaskTypes.testing,
            "script": "eval.py",
            # Pass the model representation dynamically to policy
            "argparse_args": [("policy", f"model:{m.id}")],
            "config_path": "config/eval/pusht.yaml",
            "overrides": {
                "policy": f"model:{m.id}",
                "eval.dataset_name": training_dataset_name,
                "dataset.stats": training_dataset_name,
                # Resolve the correct dataset from name
                "eval.clearml_name": "LeWM-PushT" if "push" in training_dataset_name.lower() else "LeWM-TwoRoom",
                "eval.clearml_project": "LeWM"
            },
            "tags": ["evaluation", job_id, f"epoch:{epoch}"],
            "packages": DEFAULT_EVAL_PACKAGES
        }
        enqueue_task(task_spec, queue_name)

def main():
    parser = argparse.ArgumentParser(description="Generalized ClearML Enqueue Utility for LeWM Workspace.")
    parser.add_argument("--mode", choices=["preset", "eval", "evaluate_job"], required=True,
                        help="Execution mode: run a training preset, launch evaluation on checkpoint, or evaluate full training task models.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()) + ["all_baselines"],
                        help="Name of the training preset to run (required in preset mode).")
    parser.add_argument("--checkpoint", default="lewm/weights_epoch_100.pt",
                        help="Path to checkpoint file for evaluation mode (default: lewm/weights_epoch_100.pt).")
    parser.add_argument("--all-epochs", action="store_true",
                        help="In eval mode, schedule evaluation tasks for all 100 epochs sequentially.")
    parser.add_argument("--job-id",
                        help="ClearML task ID of the training job to evaluate (required in evaluate_job mode).")
    parser.add_argument("--queue", default="default",
                        help="Target queue for execution (default: 'default').")
                        
    args = parser.parse_args()
    
    # Command Pattern mapping to dispatch modes without complex conditional nesting
    dispatch = {
        "preset": lambda: handle_preset_mode(args.preset, args.queue),
        "eval": lambda: [handle_eval_checkpoint(f"lewm/weights_epoch_{epoch}.pt", args.queue) for epoch in range(1, 101)] if args.all_epochs else handle_eval_checkpoint(args.checkpoint, args.queue),
        "evaluate_job": lambda: handle_evaluate_job(args.job_id, args.queue)
    }
    
    dispatch[args.mode]()

if __name__ == "__main__":
    main()
