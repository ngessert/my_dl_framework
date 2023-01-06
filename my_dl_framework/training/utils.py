"""
Training utility stuff
"""
import os
import json
from typing import Dict, Tuple
from torch.utils.data import Dataset
import torch
from torchvision.models import get_model
import pandas as pd
import numpy as np
import plotly
from clearml import Logger
import matplotlib.pyplot as plt
from my_dl_framework.evaluation.utils import validate_model_classification


def get_tv_class_model(config: Dict) -> torch.nn.Module:
    """
    Get a torchvision classification model and replace output layer
    :param config:          Dict with config
    :return:                Torch model
    """
    model = get_model(config["classification_model_name"], weights="DEFAULT")
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, torch.nn.Sequential):
            # Efficientnet
            in_features = model.classifier[1].in_features
            model.classifier[1] = torch.nn.Linear(in_features, config['num_classes'])
        else:
            # Densenet
            in_features = model.classifier.in_features
            model.classifier = torch.nn.Linear(in_features, config['num_classes'])
    elif hasattr(model, "fc"):
        # Resnet
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, config['num_classes'])
    elif hasattr(model, "heads"):
        # Vision transformer
        in_features = model.heads[-1].in_features
        model.heads[-1] = torch.nn.Linear(in_features, config['num_classes'])
    elif hasattr(model, "head"):
        # Swin transformer
        in_features = model.head.in_features
        model.head = torch.nn.Linear(in_features, config['num_classes'])
    else:
        raise ValueError(f'Output layer replacement not implemented for model {config["classification_model_name"]}')
    return model


def get_lossfunction(config: Dict) -> torch.nn.Module:
    """
    Get a loss function
    :param config:          Dict with config
    :return:                loss_function object
    """
    if config['loss_name'] == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss(weight=config['loss_weights'] if 'loss_weights' in config else None)
    else:
        raise ValueError(f'Unknown loss name {config["loss_name"]}')
    return loss_function


def get_optimizer(config: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Get an optimizer
    :param config:      Dict with config
    :param model:       Torch model
    :return:            optimizer object
    """
    if config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config["optimizer"] == "SGDM":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config['learning_rate'],
                                    momentum=config["sgd_momentum"])
    else:
        raise ValueError(f'Unknown optimizer name {config["optimizer"]}')
    return optimizer


def get_lr_scheduler(config: Dict, optimizer: torch.optim.Optimizer):
    """
    Get a learning rate scheduler
    :param config:          Dict with config
    :param optimizer:       Torch optimizer
    :return:                lr_scheduler object
    """
    if config['lr_scheduler'] == "stepwise":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=1/np.float32(config['lr_step']))
    else:
        raise ValueError(f'Unknown lr scheduler name {config["lr_scheduler"]}')
    return lr_scheduler


def get_and_log_metrics_classification(eval_name: str,
                                       dataloader: torch.utils.data.DataLoader,
                                       model: torch.nn.Module,
                                       config: Dict,
                                       logger: Logger,
                                       epoch: int,
                                       curr_subfolder: str,
                                       use_clearml: bool) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Function that calls the metric evaluation and then logs it ClearML and disk
    :param eval_name:       Name of the evaluation, e.g. train or test
    :param dataloader:      Dataloader to draw samples from
    :param model:           Model
    :param config:          Config dict
    :param logger:          ClearML logger
    :param epoch:           Epoch for logging
    :param curr_subfolder:  Subfolder where to store local metrics
    :param use_clearml:     Whether clearml is used
    :return:                Metrics, predictions, and targets
    """
    print(f'Validating on set {eval_name} of length {len(dataloader)}')
    metrics, plots, pred, tar = validate_model_classification(model=model, dataloader=dataloader, config=config, max_num_batches=config["max_num_batches_val"], use_cleaml=use_clearml)
    # convert to pandas DF
    metric_df = pd.DataFrame.from_dict(metrics)
    metric_df["classes"] = config["class_names"] + ["Avg"]
    metric_df.set_index("classes", inplace=True)
    print("Metrics" + "-" * 20)
    print(metric_df)
    print("-" * 25)
    if use_clearml:
        logger.report_table(title="Metrics", series="Metrics", iteration=epoch, table_plot=metric_df)
        for metric_name, metric_val in metrics.items():
            logger.report_scalar(title=metric_name, 
                                 series=eval_name + " " + metric_name,
                                 value=np.mean(metric_val),
                                 iteration=epoch)
        for plot_name, plot in plots.items():
            # Plotly figure
            if isinstance(plot, plotly.graph_objs.Figure):
                logger.report_plotly(title=eval_name + " " + plot_name, series=eval_name + " " + plot_name, figure=plot, iteration=epoch)
    # Save metrics
    if os.path.exists(os.path.join(curr_subfolder, eval_name + "_metrics.json")):
        with open(os.path.join(curr_subfolder, eval_name + "_metrics.json"), encoding="utf-8") as file:
            metrics_all = json.load(file)
        for key in metrics_all:
            metrics_all[key] = np.asarray(metrics_all[key])
        metrics_all[epoch] = metrics
    else:
        metrics_all = dict()
        metrics_all["best_epoch"] = epoch
        metrics_all[epoch] = metrics
    # Save
    with open(os.path.join(curr_subfolder, eval_name + "_metrics.json"), encoding="utf-8") as file:
        json.dump(metrics_all, file, cls=NumpyEncoder)
    return metrics_all, pred, tar


class NumpyEncoder(json.JSONEncoder):
    """ Numpy encoder for JSON saving.
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def plot_example_batch(images: torch.Tensor, targets: torch.Tensor, idx: int, save_path: str, config: Dict):
    """ Plots some example batches into clearml and to local path
    """
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 1, figsize=(5 * batch_size, 20))
    for i in range(batch_size):
        image = images[i, 0, :, :].numpy()
        target = config["class_names"][targets[i].item()]
        clearml_logger = Logger.current_logger()
        if clearml_logger is not None:
            clearml_logger.report_image(
                f"{i} of {idx} y={target}",
                "debug example",
                iteration=0,
                image=image,
            )
        axes[i].imshow(image, cmap="gray")
        axes[i].set_title(f'Target {target}', fontsize=5)
        axes[i].set_axis_off()
    os.makedirs(os.path.join(save_path, "example_batches"), exist_ok=True)
    fig.savefig(os.path.join(save_path, "example_batches", "batch_" + str(idx) + ".png"), bbox_inches='tight', pad_inches=0, dpi=300)
