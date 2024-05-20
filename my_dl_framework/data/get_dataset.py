"""
General dataset getter
"""
from typing import Dict, List, Union, Optional
from torch.utils.data import Dataset
import plotly.graph_objects as go
from my_dl_framework.data.rsna_binary.rsna_binary_dataset import RSNAChallengeBinaryDataset, plot_rsna_properties


def get_pt_dataset(config: Dict, image_dir: str, path_to_label_csv: Union[str, None], subset: Union[List[str], None], tta_options: Optional[Dict],
                   is_training: bool) -> Dataset:
    """
    Getter for a torch dataset
    :param config:          Dict with config
    :param image_dir:       Path where the images are located
    :param path_to_label_csv: Path to CSV with labels
    :param subset:          Subset to select
    :param tta_options:     TTA options dict
    :param is_training:     Whether training mode is active (e.g. for data augmentation)
    :return:                A torch dataset
    """
    if config["data"]["dataset_type"] == "RSNAChallengeBinary":
        return RSNAChallengeBinaryDataset(config=config,
                                          image_dir=image_dir,
                                          path_to_label_csv=path_to_label_csv,
                                          subset=subset,
                                          tta_options=tta_options,
                                          is_training=is_training)
    else:
        raise ValueError(f'Unknown dataset type {config["dataset_type"]}')


def get_dataset_stats(dataset_info: Dict, dataset_type: str) -> Dict[str, go.Figure]:
    """
    Creates dataset specific plots for visualizing the dataset

    :param dataset_info:        Dict with info
    :param dataset_type:        Dataset
    :return:
    """
    if dataset_type == "RSNAChallengeBinary":
        return plot_rsna_properties(dataset_info)
    else:
        raise ValueError(f'Unknown dataset type {dataset_type}')
