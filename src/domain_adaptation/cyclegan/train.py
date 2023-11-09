import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
import hydra

from src.domain_adaptation.cyclegan.cyclegan import CycleGAN
from src.datasets.unity import UnityDataset
from src.datasets.nyudv2 import NYUDv2Dataset
from src.datasets.bdsd500 import BDSD500Dataset
from src.datasets.unpaired import UnpairedDataset, UnpairedDataModule

@hydra.main(config_path="config", config_name="config", version_base="1.1.0")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(42)

    # Create model
    model = CycleGAN(input_nc_genX=cfg.model.input_nc, output_nc_genX=cfg.model.output_nc, input_nc_genY=cfg.model.output_nc, output_nc_genY=cfg.model.input_nc, log_interval=5)

    # Create a PyTorch Lightning trainer with the generation callback
    if cfg.wandb:
        logger = WandbLogger(
            name=cfg.train.name,
            project='CycleGAN',
            config=dict(cfg)
        )
    else:
        logger = None

    # profiler = pl.profilers.AdvancedProfiler(dirpath='.', filename='results.txt')
    trainer = pl.Trainer(logger=logger, max_epochs=cfg.train.epochs, devices=cfg.train.gpus, log_every_n_steps=5)

    def get_dataset(dataset_name):
        if dataset_name == "unity":
            return UnityDataset(resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size)) #, cat=(0,2))
        elif dataset_name == "nyudv2":
            return NYUDv2Dataset(cfg.data.root_nyudv2, resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size))
        elif dataset_name == "bdsd500":
            return BDSD500Dataset(cfg.data.root_bdsd500, resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size))
        else:
            raise ValueError("Dataset {} not found".format(dataset_name))

    # Create unpaired dataset
    unpaired_dataset = UnpairedDataset(get_dataset(cfg.data.dataset_a), get_dataset(cfg.data.dataset_b), mode='train')

    # Create data module
    data_module = UnpairedDataModule(
        unpaired_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=cfg.data.shuffle,
    )

    # Train the model
    trainer.fit(model, data_module.train_dataloader(), data_module.val_dataloader())

if __name__ == "__main__":
    main()
