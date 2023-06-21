import random
import torch.nn.functional as F

class Transform:
    def __init__(self, resize=False, crop_size=False, flip=True):
        self.resize = resize
        self.crop_size = crop_size
        self.flip = flip

    def __call__(self, data):
        data = list(data)

        if self.resize:
            # Apply the resize to both images
            for idx in range(len(data)):
                data[idx] = F.interpolate(data[idx].unsqueeze(0), size=self.resize, mode='bilinear', align_corners=False).squeeze(0)

        if self.crop_size:
            i, j = self.get_params(data[0], output_size=self.crop_size)

            for idx in range(len(data)):
                data[idx] = self.crop(data[idx], i, j, self.crop_size[0], self.crop_size[1])

        if self.flip:
            # Apply the random horizontal flip to both images
            if random.random() < 0.5:
                for idx in range(len(data)):
                    data[idx] = data[idx].flip(-1)

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

