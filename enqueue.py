import argparse
import os
import re
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
    "tworoom": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-tworoom-sigreg-bf16_mixed",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "tworoom")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/tworoom.yaml",
        "dataset_name": "tworoom.lance",
        "clearml_dataset_name": "LeWM-TwoRoom",
        "overrides": {
            "trainer.max_epochs": 10,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-mixed",
            "loader.batch_size": 144,
            "num_workers": 4,
            "loader.prefetch_factor": 2,
            "compile": True,
            # "trainer.limit_train_batches": 10,  # Limits training steps per epoch
            # "trainer.limit_val_batches": 2,     # Limits validation steps per epoch
        },
        "tags": ["baseline", "tworoom", "lance", "bf16-mixed", "sigreg"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "reacher": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-reacher-sigreg-bf16_mixed",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "dmc")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/dmc.yaml",
        "dataset_name": "reacher.lance",
        "clearml_dataset_name": "LeWM-Reacher",
        "overrides": {
            "trainer.max_epochs": 10,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-mixed",
            "loader.batch_size": 144,
            "num_workers": 4,
            "loader.prefetch_factor": 2,
            "compile": True,
        },
        "tags": ["baseline", "reacher", "lance", "bf16-mixed", "sigreg"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "cube": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-cube_single-sigreg-bf16_mixed",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "ogb")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/ogb.yaml",
        "dataset_name": "ogbench/cube_single_expert.lance",
        "clearml_dataset_name": "LeWM-Cube",
        "overrides": {
            "trainer.max_epochs": 10,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-mixed",
            "loader.batch_size": 144,
            "num_workers": 4,
            "loader.prefetch_factor": 2,
            "compile": True,
        },
        "tags": ["baseline", "cube_single", "lance", "bf16-mixed", "sigreg"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
    "pusht": {
        "project_name": "LeWM/Training",
        "task_name": "LeWM-Train-pusht_lance-sigreg-bf16_mixed",
        "task_type": Task.TaskTypes.training,
        "script": "train.py",
        "argparse_args": [("data", "pusht_lance")],
        "config_path": "config/train/lewm.yaml",
        "data_config_path": "config/train/data/pusht.yaml",
        "dataset_name": "pusht_expert_train.lance",
        "clearml_dataset_name": "LeWM-PushT",
        "overrides": {
            "trainer.max_epochs": 10,
            "scheduler_max_epochs": 100,
            "trainer.precision": "bf16-mixed",
            "loader.batch_size": 144,
            "loader.prefetch_factor": 2,
            "num_workers": 4,
            "compile": True,
        },
        "tags": ["baseline", "pusht", "lance", "bf16-mixed", "sigreg"],
        "packages": DEFAULT_TRAIN_PACKAGES
    },
}

def load_and_merge_configs(config_path, data_config_path=None, overrides=None):
    """Loads Hydra/OmegaConf configurations and applies custom dictionaries overrides."""
    cfg = OmegaConf.load(config_path)
    if data_config_path:
        data_cfg = OmegaConf.load(data_config_path)
        data_cfg = OmegaConf.create({"data": data_cfg})
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

def handle_preset_mode(preset_name, queue_name, precision=None):
    """Handles execution of one or multiple preset tasks using a dictionary lookup."""
    if preset_name == "all_baselines":
        # Launch baseline training targets
        targets = ["baseline_tworoom", "baseline_reacher", "baseline_cube"]
        for target in targets:
            spec = PRESETS[target].copy()
            if precision:
                spec["overrides"] = spec.get("overrides", {}).copy()
                spec["overrides"]["trainer.precision"] = precision
            enqueue_task(spec, queue_name)
    elif preset_name in PRESETS:
        spec = PRESETS[preset_name].copy()
        if precision:
            spec["overrides"] = spec.get("overrides", {}).copy()
            spec["overrides"]["trainer.precision"] = precision
        enqueue_task(spec, queue_name)
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
    """Fetches checkpoints registered by a training job and schedules evaluation tasks.

    The dataset and eval config are detected automatically from the training task's
    parameters — no assumptions are made about the environment type.
    """
    # Per-dataset lookup table: keyed by substring matched against the dataset name.
    # Order matters: more-specific substrings first.
    DATASET_SPECS = [
        {
            "match": "pusht",
            "config_path": "config/eval/pusht.yaml",
            "clearml_name": "LeWM-PushT",
        },
        {
            "match": "tworoom",
            "config_path": "config/eval/tworoom.yaml",
            "clearml_name": "LeWM-TwoRoom",
        },
        {
            "match": "reacher",
            "config_path": "config/eval/reacher.yaml",
            "clearml_name": "LeWM-Reacher",
            "extra_packages": ["mujoco", "dm-env", "dm-tree", "lxml"],
        },
        {
            "match": "cube",
            "config_path": "config/eval/cube.yaml",
            "clearml_name": "LeWM-Cube",
            "extra_packages": ["mujoco", "dm-env", "dm-tree", "lxml", "ogbench"],
        },
    ]

    print(f"🔍 Fetching training task ID: {job_id}...")
    t = Task.get_task(task_id=job_id)

    # Resolve the dataset name from task parameters (same strategy as handle_add_plot)
    training_dataset_name = t.get_parameter("General/data/dataset/name")
    if not training_dataset_name:
        training_dataset_name = t.get_parameter("General/dataset/name")
    if not training_dataset_name:
        training_dataset_name = t.get_parameter("Args/data")
    if not training_dataset_name:
        print("❌ Error: Could not determine dataset name from training task parameters.")
        sys.exit(1)

    print(f"📂 Detected dataset: {training_dataset_name}")

    # Match against the lookup table
    dataset_lower = training_dataset_name.lower()
    matched_spec = None
    for spec in DATASET_SPECS:
        if spec["match"] in dataset_lower:
            matched_spec = spec
            break

    if matched_spec is None:
        print(f"❌ Error: Dataset '{training_dataset_name}' did not match any known environment. "
              f"Supported: {[s['match'] for s in DATASET_SPECS]}")
        sys.exit(1)

    print(f"🗂  Using eval config: {matched_spec['config_path']} (ClearML dataset: {matched_spec['clearml_name']})")

    models = t.get_models().get('output', [])
    task_models = [m for m in models if m.task == job_id]

    if not task_models:
        print(f"❌ Error: No output models found registered by task {job_id}.")
        return

    print(f"📊 Found {len(task_models)} output models. Scheduling evaluation jobs...")

    for m in task_models:
        # Determine epoch from model tags
        epoch = None
        for tag in m.tags:
            if tag.startswith("epoch:"):
                try:
                    epoch = int(tag.split(":")[1])
                except ValueError:
                    pass
                break

        if epoch is None:
            # Fallback: parse from model name (e.g. "lewm-epoch-42")
            try:
                epoch = int(m.name.split("-epoch-")[1])
            except Exception:
                print(f"  Warning: Skipping model {m.id} (Name: {m.name}) - cannot determine epoch.")
                continue

        epoch_str = f"{epoch:03d}"
        task_spec = {
            "project_name": "LeWM/Evaluation",
            "task_name": f"LeWM-Eval-{epoch_str}-{t.name}",
            "task_type": Task.TaskTypes.testing,
            "script": "eval.py",
            "argparse_args": [("policy", f"model:{m.id}")],
            "config_path": matched_spec["config_path"],
            "overrides": {
                "policy": f"model:{m.id}",
                "eval.dataset_name": training_dataset_name,
                "dataset.stats": training_dataset_name,
                "eval.clearml_name": matched_spec["clearml_name"],
                "eval.clearml_project": "LeWM",
            },
            "tags": ["evaluation", job_id, f"epoch:{epoch}"],
            "packages": DEFAULT_EVAL_PACKAGES + matched_spec.get("extra_packages", []),
        }
        enqueue_task(task_spec, queue_name)

def handle_add_plot(job_id):
    """Aggregates success rates from evaluation tasks and posts a line plot onto the training task."""
    print(f"Fetching training task: {job_id}...")
    try:
        training_task = Task.get_task(task_id=job_id)
    except Exception as e:
        print(f"❌ Error fetching training task {job_id}: {e}")
        sys.exit(1)

    # Resolve the environment name from the dataset parameter
    training_dataset_name = training_task.get_parameter("General/data/dataset/name")
    if not training_dataset_name:
        training_dataset_name = training_task.get_parameter("General/dataset/name")
    if not training_dataset_name:
        training_dataset_name = training_task.get_parameter("Args/data") or "Unknown"

    dataset_lower = training_dataset_name.lower()
    if "pusht" in dataset_lower:
        env_name = "PushT"
    elif "tworoom" in dataset_lower:
        env_name = "TwoRoom"
    elif "reacher" in dataset_lower:
        env_name = "Reacher"
    elif "cube" in dataset_lower:
        env_name = "Cube"
    else:
        env_name = os.path.basename(training_dataset_name).split('.')[0].replace('_', ' ').title()

    # Find all tasks tagged with the training job ID that are also tagged 'evaluation'
    print(f"Searching for evaluation tasks tagged with: {job_id}...")
    tagged_tasks = Task.get_tasks(tags=[job_id])
    eval_tasks = [t for t in tagged_tasks if "evaluation" in t.get_tags()]
    print(f"Found {len(eval_tasks)} evaluation tasks associated with this training job.")

    epoch_to_success = {}

    for t in eval_tasks:
        # Extract epoch — try parameter first, then parse from task name
        epoch = None
        epoch_param = t.get_parameter("General/eval/epoch")
        if epoch_param:
            try:
                epoch = int(epoch_param)
            except ValueError:
                pass

        if epoch is None:
            match = re.match(r'^(\d+)_', t.name)
            if match:
                epoch = int(match.group(1))
            else:
                match = re.search(r'(?:weights_epoch_|epoch=)(\d+)', t.name)
                if match:
                    epoch = int(match.group(1))

        if epoch is None:
            print(f"  Warning: Skipping task {t.id} (Name: {t.name}) - cannot determine epoch number.")
            continue

        # Extract success rate from scalar metrics
        success_rate = None
        metrics = t.get_last_scalar_metrics()
        if metrics and "Evaluation Metrics" in metrics:
            success_rate_data = metrics["Evaluation Metrics"].get("success_rate")
            if success_rate_data:
                success_rate = success_rate_data.get("last")

        if success_rate is None:
            print(f"  Info: Skipping epoch {epoch} (Task: {t.id}, Status: {t.status}) - no 'success_rate' metrics reported yet.")
            continue

        epoch_to_success[epoch] = float(success_rate)
        print(f"  Found: Epoch {epoch} -> Success Rate: {success_rate}%")

    if not epoch_to_success:
        print("No success rate data could be collected from evaluation tasks. Plot will not be created.")
        return

    sorted_points = sorted(epoch_to_success.items())
    print(f"\nCollected {len(sorted_points)} data points for plotting:")
    for epoch, success_rate in sorted_points:
        print(f"  Epoch {epoch:3d}: {success_rate:6.2f}%")

    plot_metric = "Evaluation Performance"
    plot_variant = f"{env_name} Success Rate vs Epoch"
    print(f"\nAdding/Updating plot on training job {job_id}:")
    print(f"  Metric (Title): {plot_metric}")
    print(f"  Variant (Series): {plot_variant}")

    logger = training_task.get_logger()
    logger.report_scatter2d(
        title=plot_metric,
        series=plot_variant,
        scatter=sorted_points,
        iteration=0,
        xaxis="Epoch",
        yaxis="Success Rate"
    )
    print("✅ Successfully added plot to training task!")

def main():
    parser = argparse.ArgumentParser(description="Generalized ClearML Enqueue Utility for LeWM Workspace.")
    parser.add_argument("--mode", choices=["preset", "eval", "evaluate_job", "add_plot"], required=True,
                        help="Execution mode: run a training preset, launch evaluation on a checkpoint, evaluate all models from a training job, or aggregate eval metrics into a training task plot.")
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
    parser.add_argument("--precision",
                        help="Override numerical precision (data type) for training presets (e.g. '32' or '32-true' for FP32).")
                        
    args = parser.parse_args()
    
    # Command Pattern mapping to dispatch modes without complex conditional nesting
    dispatch = {
        "preset": lambda: handle_preset_mode(args.preset, args.queue, precision=args.precision),
        "eval": lambda: [handle_eval_checkpoint(f"lewm/weights_epoch_{epoch}.pt", args.queue) for epoch in range(1, 101)] if args.all_epochs else handle_eval_checkpoint(args.checkpoint, args.queue),
        "evaluate_job": lambda: handle_evaluate_job(args.job_id, args.queue),
        "add_plot": lambda: handle_add_plot(args.job_id)
    }

    dispatch[args.mode]()

if __name__ == "__main__":
    main()
