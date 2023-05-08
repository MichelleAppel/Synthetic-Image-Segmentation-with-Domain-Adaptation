import os
import ssl
from urllib.request import urlopen

import mat73
import numpy as np
from PIL import Image
from skimage.segmentation import find_boundaries
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from .base import DatasetBase


class NYUDv2(DatasetBase):
    def __init__(self, data_root, 
                 train_val_test_split=(0.7, 0.2, 0.1), 
                 batch_size=4, 
                 num_workers=1, 
                 *args, **kwargs):
        super().__init__(data_root, *args, **kwargs)
        self.filename = "nyu_depth_v2_labeled.mat"
        self.url = "https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
        self.train_val_test_split = train_val_test_split
        self.input_height = 480
        self.input_width = 640
        self.batch_size = batch_size
        self.num_workers = num_workers

    def download(self, verify_ssl=False):
        # Check if the file is already downloaded
        file_path = os.path.join(self.data_root, self.filename)
        if not os.path.exists(file_path):
            # Download the .mat file
            ssl_context = None
            if not verify_ssl:
                ssl_context = ssl._create_unverified_context()

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Get file size from the header
            with urlopen(self.url, context=ssl_context) as response:
                file_size = int(response.info().get('Content-Length', 0))

            # Download the file with a progress bar
            with urlopen(self.url, context=ssl_context) as response, open(file_path, "wb") as out_file:
                chunk_size = 1024
                total_chunks = (file_size + chunk_size - 1) // chunk_size
                for _ in tqdm(range(total_chunks), total=total_chunks, unit='KB'):
                    chunk = response.read(chunk_size)
                    out_file.write(chunk)

    def prepare_data(self):
        # Load the .mat file
        file_path = os.path.join(self.data_root, self.filename)
        data = mat73.loadmat(file_path)

        # Extract the images and instances
        images = data["images"]
        instances = data["instances"]

        # Create directories for images, instances, and edges if they don't exist
        os.makedirs(os.path.join(self.data_root, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.data_root, "edges"), exist_ok=True)

        # Save the images, instances, and edges as .png files
        for i in range(images.shape[3]):
            # Save image
            image = Image.fromarray(images[:, :, :, i])
            image.save(os.path.join(self.data_root, "images", f"{i:05d}.png"))

            # Save instance
            instance = Image.fromarray(instances[:, :, i])

            # Compute and save edges
            instance_np = np.array(instance)
            edges = self.compute_edges(instance_np)
            edges_img = Image.fromarray(edges.astype(np.uint8) * 255)
            edges_img.save(os.path.join(self.data_root, "edges", f"{i:05d}.png"))

    def compute_edges(self, instance):
        edges = find_boundaries(instance, mode='outer').astype(np.uint8)
        return edges

    def setup(self):
        # Load the dataset if not already loaded
        if not os.path.exists(os.path.join(self.data_root, "images")):
            self.prepare_data()

        # Generate image and edge paths
        image_dir = os.path.join(self.data_root, "images")
        edges_dir = os.path.join(self.data_root, "edges")

        image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
        edges_paths = sorted([os.path.join(edges_dir, f) for f in os.listdir(edges_dir) if f.endswith(".png")])

        # Split the dataset into train, validation, and test sets
        num_samples = len(image_paths)
        train_size = int(self.train_val_test_split[0] * num_samples)
        val_size = int(self.train_val_test_split[1] * num_samples)

        self.train_image_paths = image_paths[:train_size]
        self.train_edges_paths = edges_paths[:train_size]

        self.val_image_paths = image_paths[train_size:train_size + val_size]
        self.val_edges_paths = edges_paths[train_size:train_size + val_size]

        self.test_image_paths = image_paths[train_size + val_size:]
        self.test_edges_paths = edges_paths[train_size + val_size:]

    def train_dataloader(self):
            train_dataset = self._create_dataset(self.train_image_paths, self.train_edges_paths)
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = self._create_dataset(self.val_image_paths, self.val_edges_paths)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = self._create_dataset(self.test_image_paths, self.test_edges_paths)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def _create_dataset(self, image_paths, segmentation_paths):
        transform = Compose([
            Resize((self.input_height, self.input_width)),
            ToTensor(),
        ])

        dataset = NYUDv2Dataset(image_paths=image_paths, segmentation_paths=segmentation_paths, transform=transform)
        return dataset

class NYUDv2Dataset(Dataset):
    ''' Dataset class for the NYUDv2 dataset '''
    def __init__(self, image_paths, segmentation_paths, transform=None):
        self.image_paths = image_paths # List of image paths
        self.segmentation_paths = segmentation_paths # List of segmentation paths
        self.transform = transform # Transform to apply to the images and segmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        segmentation_path = self.segmentation_paths[idx]
        segmentation = Image.open(segmentation_path)

        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)

        return image, segmentation
