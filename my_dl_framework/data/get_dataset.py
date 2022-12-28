"""
General dataset getter
"""
from typing import Dict, List, Union
from torch.utils.data import Dataset

from my_dl_framework.data.rsna_binary.rsna_binary_dataset import RSNAChallengeBinaryDataset


def get_dataset(config: Dict, image_dir: str, subset: Union[List[str], None], is_training: bool) -> Dataset:
    """
    Getter for a torch dataset
    :param config:          Dict with config
    :param image_dir:       Path where the images are located
    :param subset:          Subset to select
    :param is_training:     Whether training mode is active (e.g. for data augmentation)
    :return:                A torch dataset
    """
    if config["dataset_type"] == "RSNAChallengeBinary":
        return RSNAChallengeBinaryDataset(config=config, image_dir=image_dir, subset=subset, is_training=is_training)
    else:
        raise ValueError(f'Unknown dataset type {config["dataset_type"]}')