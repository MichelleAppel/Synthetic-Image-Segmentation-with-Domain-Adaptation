import socket
import io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import struct
import threading

from torch import cat

from src.datasets.transforms import Transform

from torch.utils.data.sampler import SubsetRandomSampler

class UnityDataset(Dataset):
    def __init__(self, host="127.0.0.1", port=8093, epoch_length=10000, size=(480, 640), cat=False):
        self.host = host
        self.port = port
        self.epoch_length = epoch_length
        self.n_cameras = 2
        self.size = size
        self.transforms = Transform(self.size)
        self.cat = False

    def worker_init_fn(self, worker_id):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))

    def __del__(self):
        # Make sure to close the socket connection when the dataset is deleted
        self.socket.close()

    def __len__(self):
        # Define an arbitrary length of one "epoch"
        return self.epoch_length

    def __getitem__(self, index):
        # Convert index to string and send it as bytes
        index_string = str(index)
        self.socket.sendall(index_string.encode('utf-8'))

        data = self._receive_images()

        # Apply transforms
        data = self.transforms(data)

        if self.cat:
            data = cat(data, dim=0)

        return data

    def _receive_images(self):
        images = []

        # First, receive the number of cameras
        num_cameras_data = self.socket.recv(4)
        num_cameras = struct.unpack('!I', num_cameras_data)[0]  # Network byte order is big endian

        for _ in range(num_cameras):
            # Receive the length of the image data
            length_data = self.socket.recv(4)
            if not length_data:
                break  # No more data to receive

            length = struct.unpack('!I', length_data)[0]  # Network byte order is big endian

            # Now receive the image data
            received_data = b''
            while len(received_data) < length:
                remaining_bytes = length - len(received_data)
                data = self.socket.recv(4096 if remaining_bytes > 4096 else remaining_bytes)
                if not data:
                    break  # No more data to receive
                received_data += data

            if len(received_data) < length:
                break  # Incomplete data received

            image = Image.open(io.BytesIO(received_data))

            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode == 'LA':
                image = image.convert('L')

            images.append(image)

            marker = self.socket.recv(3)
            if marker == b'EOI':
                continue

        # Check for end of transmission
        marker = self.socket.recv(3)
        if marker == b'EOT':
            pass

        return images

class UnityDataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=1, num_workers=0, shuffle=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def train_dataloader(self):
        indices = list(range(len(self.dataset)))
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=SubsetRandomSampler(indices), shuffle=False, worker_init_fn=self.dataset.worker_init_fn)

    def val_dataloader(self):
        indices = list(range(len(self.dataset), 2 * len(self.dataset)))
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=SubsetRandomSampler(indices), shuffle=False, worker_init_fn=self.dataset.worker_init_fn)

    def test_dataloader(self):
        indices = list(range(2 * len(self.dataset), 3 * len(self.dataset)))
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=SubsetRandomSampler(indices), shuffle=False, worker_init_fn=self.dataset.worker_init_fn)