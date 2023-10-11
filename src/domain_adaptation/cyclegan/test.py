import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    model.load_from_checkpoint(cfg.test.checkpoint_path)

    # select datasets from config
    wandb_logger = WandbLogger(project='CycleGAN test', config=cfg, name=cfg.train.name)
    wandb_logger.watch(model)

    # profiler = pl.profilers.AdvancedProfiler(dirpath='.', filename='results.txt')
    trainer = pl.Trainer(logger=wandb_logger)

    datasets = {"unity": UnityDataset(resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size), cat=(0,2)),
                "nyudv2": NYUDv2Dataset(cfg.data.root_nyudv2, resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size)),
                "bdsd500": BDSD500Dataset(cfg.data.root_bdsd500, resize=tuple(cfg.data.resize), crop_size=tuple(cfg.data.crop_size))}

    # Create unpaired dataset
    unpaired_dataset = UnpairedDataset(datasets[cfg.data.dataset_a], datasets[cfg.data.dataset_b], mode='test')

    # Create data module
    data_module = UnpairedDataModule(
        unpaired_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=cfg.data.shuffle,
    )

    # Train the model
    results = trainer.predict(model, datamodule=data_module)
    print(results)

if __name__ == "__main__":
    main()
