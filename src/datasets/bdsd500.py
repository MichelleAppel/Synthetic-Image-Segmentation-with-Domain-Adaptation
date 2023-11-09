import os
from urllib.request import urlretrieve

import numpy as np
import scipy
from PIL import Image
from skimage.segmentation import find_boundaries
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from torchvision.io import read_image
import torch

from src.datasets.transforms import Transform

import pytorch_lightning as pl

class BDSD500Dataset(Dataset):
    def __init__(self, data_root, resize=None, crop_size=(321, 321)):

        self.data_root = data_root
        self.filename = "BSR_bsds500.tgz"
        self.url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"

        self.resize = resize
        self.crop_size = crop_size

        self.transforms = Transform(resize=self.resize, crop_size=self.crop_size)

        self.image_paths = []
        self.edges_paths = []

        self.setup()

    def download(self):
        # Check if the file is already downloaded
        file_path = os.path.join(self.data_root, self.filename)
        if not os.path.exists(file_path):
            # Create the directory if it doesn't exist
            os.makedirs(self.data_root, exist_ok=True)

            # Download the file with a progress bar
            urlretrieve(self.url, file_path, self._progress_bar)

    def _progress_bar(self, count, block_size, total_size):
        progress = count * block_size / total_size * 100
        print(f"\rDownloading: {progress:.2f}%", end="")

    def _unzip(self, file_path, extract_dir):
        import tarfile

        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)

    def prepare_data(self):
        # Unzip the file
        self._unzip(os.path.join(self.data_root, self.filename), self.data_root)

        # Define file paths
        image_dir = os.path.join(self.data_root, "BSR", "BSDS500", "data", "images")
        gt_dir = os.path.join(self.data_root, "BSR", "BSDS500", "data", "groundTruth")
        image_save_dir = os.path.join(self.data_root, "images")
        edges_save_dir = os.path.join(self.data_root, "edges")

        # Create directories for images and edges if they don't exist
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(edges_save_dir, exist_ok=True)

        # Loop through the train, val, and test sets
        for set_name in ["train", "val", "test"]:
            # Define the path to the current set
            set_dir = os.path.join(image_dir, set_name)

            # Loop through the images in the current set
            for image_name in tqdm(os.listdir(set_dir), desc=f"Processing {set_name} set"):
                if not image_name.endswith(".jpg"):
                    continue

                # Define the path to the current image and ground truth
                image_path = os.path.join(set_dir, image_name)
                gt_path = os.path.join(gt_dir, set_name, image_name.replace(".jpg", ".mat"))

                # Load the image and ground truth
                image = Image.open(image_path)
                gt_data = scipy.io.loadmat(gt_path)
                gt = gt_data["groundTruth"][0][0][0][0][0]

                # Save the image
                image.save(os.path.join(image_save_dir, image_name))

                # Compute and save edges
                edges = self.compute_edges(gt)
                edges_img = Image.fromarray(edges.astype(np.uint8) * 255)
                edges_img.save(os.path.join(edges_save_dir, image_name.replace(".jpg", ".png")))


    def compute_edges(self, instance):
        edges = find_boundaries(instance, mode='outer').astype(np.uint8)
        return edges
    
    def setup(self):
        # Download the dataset if the path doesn't exist
        if not os.path.exists(os.path.join(self.data_root, self.filename)):
            self.download()

        # Load the dataset if not already loaded
        if not os.path.exists(os.path.join(self.data_root, "images")):
            self.prepare_data()

        # Define file paths
        image_dir = os.path.join(self.data_root, "images")
        edges_dir = os.path.join(self.data_root, "edges")

        # Get all image and edge paths
        self.image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
        self.edges_paths = [os.path.join(edges_dir, edge_name) for edge_name in os.listdir(edges_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Open the image file
        image_path = self.image_paths[idx]
        image = read_image(image_path).to(torch.float32) / 255.0 * 2 - 1

        # Open the edges file
        edges_path = self.edges_paths[idx]
        edges = read_image(edges_path).to(torch.float32) / 255.0

        data = [image, edges]

        # Apply transforms
        if self.transforms:
            data = self.transforms(data)

        return data


class BDSD500DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=4, num_workers=1, shuffle=True, split=(0.7, 0.1, 0.2)):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.dataset = dataset
        self.split = split

        self.setup()

    def setup(self):
        train_len = int(self.split[0] * len(self.dataset))
        val_len = int(self.split[1] * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_len, val_len, test_len])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)