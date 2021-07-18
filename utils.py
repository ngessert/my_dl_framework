"""
Training utility stuff
"""
from typing import Dict
from torch.utils.data import Dataset


def get_dataset(config: Dict) -> Dataset:
    if config["dataset_type"] == "RSNAChallenge":
        return RSNAChallengeDataset(config=config)
    else:
        raise ValueError(f'Unknown dataset type {config["dataset_type"]}')


class RSNAChallengeDataset(Dataset):
    def __init__(self, config: Dict):
        self.config = config

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = None
        label = None
        return image, label
