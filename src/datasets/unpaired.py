from torch.utils.data import Dataset
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import vstack

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(480),
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.5, .5, .5], [.5, .5, .5])
    ]),
}

class UnpairedDataset(Dataset):
    def __init__(self, dataset_A, dataset_B, mode, fuse_A = None, fuse_B = None):
        """
        dataset_A: instance of a Dataset
        dataset_B: instance of a Dataset
        mode must be either 'train' or 'test'
        """
        assert mode in 'train test'.split(), 'mode should be either train or test'
        
        super().__init__()
        self.transforms = data_transforms[mode]
        
        self.dirA = dataset_A
        self.dirB = dataset_B

        print(f'Found {len(self.dirA)} images of {mode}A and {len(self.dirB)} images of {mode}B')

        self.fuse_A = fuse_A
        self.fuse_B = fuse_B
        
    def __len__(self):
        return max(len(self.dirA), len(self.dirB))
    
    def __getitem__(self, idx):
        idxA = idx % len(self.dirA)  # Use modulo to prevent index out of range
        idxB = idx % len(self.dirB)  # Use modulo to prevent index out of range

        if self.fuse_A:
            imgA = vstack([self.dirA[idxA][i] for i in self.fuse_A]) # Stack the images from dataset_A
        else:
            imgA = self.dirA[idxA][0]  # Get the image from dataset_A

        if self.fuse_B:
            imgB = vstack([self.dirB[idxB][i] for i in self.fuse_B]) # Stack the images from dataset_B
        else:
            imgB = self.dirB[idxB][0]  # Get the image from dataset_B

        return {
            'A': imgA, 
            'B': imgB
        }

    
class UnpairedDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=1, num_workers=0, shuffle=True, split=(0.7, 0.15, 0.15)):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.split = split

        self.setup()

    def setup(self):
        train_len = int(self.split[0] * len(self.dataset))
        val_len = int(self.split[1] * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len

        # train_len = 64 # int(self.split[0] * self.epoch_length)
        # val_len = 0 # int(self.split[1] * self.epoch_length)
        # test_len = len(self.dataset) - train_len - val_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)