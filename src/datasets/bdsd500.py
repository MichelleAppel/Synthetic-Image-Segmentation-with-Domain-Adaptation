import os
import ssl
from urllib.request import urlretrieve

import numpy as np
import scipy
from PIL import Image
from skimage.segmentation import find_boundaries
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

from base import DatasetBase


class BDSD500(DatasetBase):
    def __init__(self, data_root, 
                 train_val_test_split=(0.6, 0.2, 0.2), 
                 batch_size=4, 
                 num_workers=1, 
                 *args, 
                 **kwargs):
        super().__init__(data_root, *args, **kwargs)
        self.filename = "BSR_bsds500.tgz"
        self.url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
        self.train_val_test_split = train_val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers

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
        # Define file paths
        image_dir = os.path.join(self.data_root, "images")
        edges_dir = os.path.join(self.data_root, "edges")

        # Get all image and edge paths
        image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
        edge_paths = [os.path.join(edges_dir, image_name.replace(".jpg", ".png")) for image_name in os.listdir(image_dir)]

        # Shuffle the paths
        image_paths = np.array(image_paths)
        edge_paths = np.array(edge_paths)
        indices = np.arange(len(image_paths))
        np.random.shuffle(indices)
        image_paths = image_paths[indices]
        edge_paths = edge_paths[indices]

        # Split the paths into train, val, and test sets
        train_val_split = int(len(image_paths) * self.train_val_test_split[0])
        val_test_split = int(len(image_paths) * (self.train_val_test_split[0] + self.train_val_test_split[1]))

        self.train_image_paths = image_paths[:train_val_split]
        self.train_edges_paths = edge_paths[:train_val_split]
        self.val_image_paths = image_paths[train_val_split:val_test_split]
        self.val_edges_paths = edge_paths[train_val_split:val_test_split]
        self.test_image_paths = image_paths[val_test_split:]
        self.test_edges_paths = edge_paths[val_test_split:]

    def train_dataloader(self):
        train_dataset = self._create_dataset(self.train_image_paths, self.train_edges_paths)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        val_dataset = self._create_dataset(self.val_image_paths, self.val_edges_paths)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        test_dataset = self._create_dataset(self.test_image_paths, self.test_edges_paths)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
    
    def _create_dataset(self, image_paths, edge_paths):
        return BDSD500Dataset(image_paths, edge_paths)

class BDSD500Dataset(Dataset):
    def __init__(self, image_paths, segmentation_paths, transform=None):
        self.image_paths = image_paths
        self.segmentation_paths = segmentation_paths
        self.transform = transform
        self.to_tensor = ToTensor()  # Add this line

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        segmentation_path = self.segmentation_paths[idx]
        with Image.open(segmentation_path) as seg:
            segmentation = seg.convert('L')  # convert to grayscale

        if self.transform:
            image = self.transform(image)
            segmentation = self.transform(segmentation)

        return self.to_tensor(image), self.to_tensor(segmentation)


