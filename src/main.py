import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from omegaconf import DictConfig
import hydra
from hydra.utils import instantiate

from src.domain_adaptation import base as da_base
from src.segmentation import base as seg_base
from src.datasets import base as ds_base
from src.utils import hydra_utils

@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(cfg.pretty())

    # Set up logging with WandB
    wandb_logger = WandbLogger(project="AI_Research_Project", config=cfg)

    # Set up the data module
    data_module = instantiate(cfg.dataset)
    data_module.prepare_data()
    data_module.setup()

    # Set up the domain adaptation model and train if necessary
    domain_adaptation_model = None
    if cfg.train.domain_adaptation:
        domain_adaptation_model = instantiate(cfg.domain_adaptation)
        da_trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=cfg.train.domain_adaptation_epochs,
            gpus=hydra_utils.get_num_gpus(cfg),
            deterministic=True,
            check_val_every_n_epoch=cfg.train.domain_adaptation_val_check_interval,
        )
        da_trainer.fit(domain_adaptation_model, datamodule=data_module)

    # Set up the segmentation model and train if necessary
    if cfg.train.segmentation:
        segmentation_model = instantiate(cfg.segmentation, domain_adaptation_model=domain_adaptation_model)
        seg_trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=cfg.train.segmentation_epochs,
            gpus=hydra_utils.get_num_gpus(cfg),
            deterministic=True,
            check_val_every_n_epoch=cfg.train.segmentation_val_check_interval,
        )
        seg_trainer.fit(segmentation_model, datamodule=data_module)

if __name__ == "__main__":
    main()