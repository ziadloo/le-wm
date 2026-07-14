import os
import ssl
import urllib3
import re
import argparse

urllib3.disable_warnings()
os.environ['CLEARML_API_HOST_VERIFY_CERT'] = '0'
os.environ['CLEARML_FILES_HOST_VERIFY_CERT'] = '0'
os.environ['CLEARML_WEB_HOST_VERIFY_CERT'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

from clearml import Task


def add_eval_plot(training_job_id):
    # Fetch the training task
    print(f"Fetching training task: {training_job_id}...")
    try:
        training_task = Task.get_task(task_id=training_job_id)
    except Exception as e:
        print(f"Error fetching training task {training_job_id}: {e}")
        return

    # Determine environment/dataset name for the plot title.
    # The new clearml-branch config nests dataset under data/dataset.
    training_dataset_name = training_task.get_parameter("General/data/dataset/name")
    if not training_dataset_name:
        training_dataset_name = training_task.get_parameter("General/dataset/name")
    if not training_dataset_name:
        training_dataset_name = training_task.get_parameter("Args/data") or "Unknown"

    # Format a human-readable environment name from the dataset path/name
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

    # Search for all tasks tagged with the training job ID
    print(f"Searching for evaluation tasks tagged with: {training_job_id}...")
    tagged_tasks = Task.get_tasks(tags=[training_job_id])

    # Keep only tasks also tagged as 'evaluation'
    eval_tasks = [t for t in tagged_tasks if "evaluation" in t.get_tags()]
    print(f"Found {len(eval_tasks)} evaluation tasks associated with this training job.")

    epoch_to_success = {}

    for t in eval_tasks:
        # 1. Extract epoch number — try parameter first, then parse from task name
        epoch = None
        epoch_param = t.get_parameter("General/eval/epoch")
        if epoch_param:
            try:
                epoch = int(epoch_param)
            except ValueError:
                pass

        if epoch is None:
            # Patterns: '02_LeWM-...' or 'weights_epoch_2' or 'epoch=2'
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

        # 2. Extract success rate from scalar metrics
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

    # Sort data points by epoch for a sequential line
    sorted_points = sorted(epoch_to_success.items())
    print(f"\nCollected {len(sorted_points)} data points for plotting:")
    for epoch, success_rate in sorted_points:
        print(f"  Epoch {epoch:3d}: {success_rate:6.2f}%")

    plot_metric = "Evaluation Performance"
    plot_variant = f"{env_name} Success Rate vs Epoch"

    print(f"\nAdding/Updating plot on training job {training_job_id}:")
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

    print("Successfully added plot to training task!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract success rate metrics from evaluation tasks and add a line plot to the training task."
    )
    parser.add_argument("training_job_id", help="ClearML Task ID of the training task to add the plot to.")
    args = parser.parse_args()

    add_eval_plot(args.training_job_id)
