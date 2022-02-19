import torch
import numpy as np
from typing import Dict
from torch.utils.data import DataLoader


def validate_model(model: torch.nn.Module, dataloader: DataLoader, config: Dict) -> Dict:
    """
    Validates a model using data from a dataloader
    :param model:           Torch model
    :param dataloader:      Dataloader
    :param config:          Dict with config
    :return:                Metrics
    """
    metrics = dict()
    model.eval()


    return metrics
