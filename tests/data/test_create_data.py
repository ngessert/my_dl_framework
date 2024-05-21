from clearml import Dataset

from my_dl_framework.data.create_dataset import create_dataset


def test_create_dataset():
    dataset_id = create_dataset('tests/test_configs/test_config.yaml')
    Dataset.delete(dataset_id=dataset_id)
