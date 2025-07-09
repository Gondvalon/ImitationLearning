from pathlib import Path

import hydra
from hydra.utils import get_original_cwd

import wandb
import os
import time

from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf, open_dict

from torch.utils.data import DataLoader, random_split, Subset
from imitation_learning_dataset import ImitationLearningDataset

from il_training import ILTrainingLightning

import pytorch_lightning as L
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger

from scripts.logging_utils import find_git_root_from_file

import torch

torch.set_float32_matmul_precision("high")


def init_dataloaders(dataset_path, batch_size, num_workers, validation_split, **kwargs):
    # Create list of files with trajectories to train the network
    filepaths_l = []
    for file_path in os.listdir(dataset_path):
        if file_path.endswith(".h5"):
            filepaths_l.append(Path(dataset_path, file_path))

    assert len(filepaths_l) > 0, f"No dataset is in {dataset_path}"

    dataset = ImitationLearningDataset(
        filepaths_l=filepaths_l,
    )
    print(f"Total dataset size: {len(dataset)}")

    num_samples = len(dataset)
    num_val = int(num_samples * validation_split)
    num_train = num_samples - num_val
    print(f"Training samples: {num_train}, Validation samples: {num_val}")

    train_dataset, val_dataset = random_split(dataset, [num_train, num_val])

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # Do not shuffle validation data, to better assess performance
        pin_memory=True,
        drop_last=True,
    )

    return train_dataloader, val_dataloader


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    root_dir = find_git_root_from_file(__file__)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(root_dir, exist_ok=True)

    cfg.wandb_il.dir = str(root_dir / cfg.wandb_il.dir)
    cfg.training.logging_dir = str(root_dir / cfg.training.logging_dir / timestamp)
    os.makedirs(cfg.wandb_il.dir, exist_ok=True)
    os.makedirs(cfg.training.logging_dir, exist_ok=True)

    # Fix seed for reproducibility
    L.seed_everything(cfg.training.seed, workers=True)

    # Initialize the dataloaders
    train_dataloader, validation_dataloader = init_dataloaders(
        **cfg.data, **cfg.training
    )

    # Initialize the PyTorch Lightning Trainer
    lightning_model = ILTrainingLightning(**cfg.training)

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.device,
        devices=1 if cfg.training.device == "gpu" else cfg.training.cpu_cores,
        log_every_n_steps=10,
        logger=L.loggers.WandbLogger(
            mode=cfg.wandb_il.wandb_mode,
            project=cfg.wandb_il.project,
            entity=cfg.wandb_il.entity,
            name=f"IL_{int(time.time())}",
            save_dir=cfg.wandb_il.dir,
            config={
                "learning_rate": cfg.training.learning_rate,
                "batch_size": cfg.training.batch_size,
                "num_epochs": cfg.training.max_epochs,
            },
        ),
        strategy=DDPStrategy(
            find_unused_parameters=True
        ),  # Enable unused parameter detection
        fast_dev_run=False,
    )

    # Train the model
    trainer.fit(lightning_model, train_dataloader, validation_dataloader)

    # Clean up
    wandb.finish()


if __name__ == "__main__":
    main()
