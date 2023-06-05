import matplotlib.pyplot as plt

from data.synthetic.unity_dataset import UnityDataset

def test_UnityDataset():
    # Create an instance of UnityDataset
    dataset = UnityDataset()

    # Try to get an image from the dataset
    image = dataset[0]

    # If an image is returned, display it using matplotlib
    assert image is not None, "Failed to retrieve image from UnityDataset"

    print(image.shape)
