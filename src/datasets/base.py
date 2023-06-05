import os
import pytorch_lightning as pl

from pytorch_lightning import seed_everything

class DatasetBase(pl.LightningDataModule):
    def __init__(self, data_root, seed=42, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_root = data_root 
        seed_everything(seed) # Set seed for reproducibility

    def download(self):
        # Download dataset if necessary
        pass

    def prepare_data(self):
        # Prepare data (e.g., unzip files, convert file formats, etc.)
        pass

    def setup(self):
        # Split data into train, validation, and test sets
        pass

    def train_dataloader(self):
        # Return a DataLoader for the training set
        pass

    def val_dataloader(self):
        # Return a DataLoader for the validation set
        pass

    def test_dataloader(self):
        # Return a DataLoader for the test set
        pass