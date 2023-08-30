import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
import hydra

from src.domain_adaptation.cyclegan.cyclegan import CycleGAN
from src.datasets.unity import UnityDataset, UnityDataModule
from src.datasets.nyudv2 import NYUDv2Dataset, NYUDv2DataModule
from src.datasets.bdsd500 import BDSD500Dataset, BDSD500DataModule

from src.segmentation.models.bdcn import BDCN

@hydra.main(config_path="config", config_name="config", version_base="1.1.0")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    # Create model
    model = BDCN()

    # Create a PyTorch Lightning trainer with the generation callback
    wandb_logger = WandbLogger(project='Segmentation', config=cfg, name=cfg.train.name)
    wandb_logger.watch(model)

    # profiler = pl.profilers.AdvancedProfiler(dirpath='.', filename='results.txt')
    trainer = pl.Trainer(logger=wandb_logger, max_epochs=cfg.train.epochs, devices=cfg.train.gpus, log_every_n_steps=5)

    # Create datasets
    # dataset = BDSD500Dataset(cfg.data.root_bdsd500, resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size))
    # dataset = UnityDataset(resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size)) #, cat=(0,2))
    dataset = NYUDv2Dataset(cfg.data.root_nyudv2, resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size))
    data_module = NYUDv2DataModule(dataset, batch_size=cfg.data.batch_size)

    # Train the model
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader()) #, ckpt_path = r"C:\Users\appel\Documents\Project\synthetic-image-segmentation\outputs\2023-08-01\16-10-02\CycleGAN\sazmzqw3\checkpoints\epoch=348-step=11168.ckpt")

if __name__ == "__main__":
    main()
