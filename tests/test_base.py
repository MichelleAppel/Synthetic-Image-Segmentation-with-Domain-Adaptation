import tempfile
from src.datasets.base import DatasetBase

def test_init():
    with tempfile.TemporaryDirectory() as temp_data_root:
        dataset = DatasetBase(temp_data_root)
        assert dataset.data_root == temp_data_root

