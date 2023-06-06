from torchvision.transforms import functional as F
from torchvision import transforms
import random

class Transform:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        data = list(data)
        i, j, h, w = transforms.RandomCrop.get_params(data[0], output_size=self.size)

        for idx in range(len(data)):
            data[idx] = F.crop(data[idx], i, j, h, w)
            data[idx] = F.to_tensor(data[idx])

        # Apply the random horizontal flip to both images
        if random.random() < 0.5:
            for idx in range(len(data)):
                data[idx] = F.hflip(data[idx])

        return data
