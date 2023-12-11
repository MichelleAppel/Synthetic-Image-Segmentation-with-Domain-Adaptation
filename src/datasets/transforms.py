import random
import torch.nn.functional as F
from torchvision import transforms

class Transform:
    def __init__(self, resize=False, crop_size=False, flip=True, normalize=False):
        self.resize = resize
        self.crop_size = crop_size
        self.flip = flip
        self.normalize = normalize

    def __call__(self, data):
        data = list(data)

        if self.resize:
            # Apply the resize to both images
            for idx in range(len(data)):
                data[idx] = F.interpolate(data[idx].unsqueeze(0), scale_factor=1, mode='bilinear', align_corners=False).squeeze(0)

        if self.crop_size:
            i, j = self.get_params(data[0], output_size=self.crop_size)

            for idx in range(len(data)):
                data[idx] = self.crop(data[idx], i, j, self.crop_size[0], self.crop_size[1])

        if self.flip:
            # Apply the random horizontal flip to both images
            if random.random() < 0.5:
                for idx in range(len(data)):
                    data[idx] = data[idx].flip(-1)

        if self.normalize:
            # Apply to just the input image
            data[0] = self.normalize(data[0])

        return data

    @staticmethod
    def get_params(img, output_size):
        w, h = img.shape[-2:]
        th, tw = output_size
        if w < tw or h < th:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        i = random.randint(0, w - tw)
        j = random.randint(0, h - th)
        return i, j

    @staticmethod
    def crop(img, i, j, h, w):
        return img[..., i:i + w, j:j + h]
    
    @staticmethod
    def normalize(img):
        n_dims = img.shape[0]
        if n_dims == 3:
            return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        elif n_dims == 4:
            return transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.5], std=[0.229, 0.224, 0.225, 0.5])(img)
        if n_dims == 1:
            return transforms.Normalize(mean=[0.5], std=[0.5])(img)
        else:
            raise ValueError(f"Unsupported number of dimensions {n_dims}")
        


