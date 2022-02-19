import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import DataLoader
from tqdm import tqdm


def validate_model_classification(model: torch.nn.Module, dataloader: DataLoader, config: Dict, max_num_batches: int, class_names: List[str]) -> (Dict, List):
    """
    Validates a model using data from a dataloader
    :param model:           Torch model
    :param dataloader:      Dataloader
    :param config:          Dict with config
    :param max_num_batches: Maximum number of batches to use
    :param class_names:     Class names
    :return:                Metrics
    """
    metrics = dict()
    plots = list()
    model.eval()
    # Get predictions
    all_predictions = dict()
    all_targets = dict()
    for batch_idx, (indices, images, targets) in tqdm(enumerate(dataloader)):
        if max_num_batches == batch_idx:
            break
        images = images.cuda()
        targets = targets.cuda()
        outputs = model(images)
        for idx, prediction, target in zip(indices, targets, outputs):
            if idx not in all_predictions:
                all_predictions[idx] = list()
            if idx not in all_targets:
                all_targets[idx] = list()
            all_predictions[idx].append(prediction)
            all_targets[idx].append(target)
    # Put into proper arrays
    all_predictions_arr = np.zeros([len(all_predictions), len(class_names)])
    all_targets_arr = np.zeros([len(all_predictions)])
    for idx, key in enumerate(all_predictions):
        if config["test_aug_ensemble_mode"] == "mean":
            all_predictions_arr[idx, :] = np.mean(all_predictions[key])
        else:
            all_predictions_arr[idx, :] = all_predictions_arr[key][0]
        all_targets_arr[idx] = all_targets[key]
    # Metrics and plots
    for metric in config["validation_metrics"]:
        if metric == "accuracy":
            pass
        if metric == "AUC":
            pass
    return metrics, plots
