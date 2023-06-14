import random

class Transform:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        data = list(data)
        i, j = self.get_params(data[0], output_size=self.size)

        for idx in range(len(data)):
            data[idx] = self.crop(data[idx], i, j, self.size[0], self.size[1])

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

