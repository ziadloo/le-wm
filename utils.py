import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1, max_epochs: int = None):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval
        self.max_epochs = max_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)
        
        if trainer.is_global_zero:
            epoch = trainer.current_epoch + 1
            if epoch % self.epoch_interval == 0 or (trainer.max_epochs and epoch == trainer.max_epochs):
                self._save(pl_module.model, epoch)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )

        from clearml import Task
        task = Task.current_task()
        if task:
            from clearml import OutputModel
            from stable_worldmodel.data.utils import get_cache_dir
            from pathlib import Path
            ckpt_dir = Path(get_cache_dir(sub_folder='checkpoints')) / self.run_name
            ckpt_path = ckpt_dir / f'weights_epoch_{epoch}.pt'
            
            if ckpt_path.exists():
                print(f"📦 Registering OutputModel with ClearML for epoch {epoch}...")
                output_model = OutputModel(task=task, name=f"{self.run_name}-epoch-{epoch}")
                output_model.update_weights(weights_filename=str(ckpt_path))
                
                from omegaconf import OmegaConf, DictConfig
                if self.cfg is not None:
                    cfg_dict = OmegaConf.to_container(self.cfg, resolve=True) if isinstance(self.cfg, DictConfig) else self.cfg
                    if isinstance(cfg_dict, dict):
                        output_model.update_design(config_dict=cfg_dict)
                
                tags = [f"epoch:{epoch}"]
                if self.max_epochs and epoch == self.max_epochs:
                    output_model.publish()
                    tags.append("final-checkpoint")
                output_model.tags = tags