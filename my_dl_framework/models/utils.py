"""
Model-related utils
"""
from typing import Dict
import torch
from torchvision.models import get_model
import numpy as np


def get_tv_class_model(config: Dict) -> torch.nn.Module:
    """
    Get a torchvision classification model and replace output layer
    :param config:          Dict with config
    :return:                Torch model
    """
    model = get_model(config["classification_model_name"], weights="DEFAULT")
    if hasattr(model, "classifier"):
        if isinstance(model.classifier, torch.nn.Sequential):
            # Squeezenet
            if len(model.classifier) == 4:
                final_conv = torch.nn.Conv2d(model.classifier[1].in_channels, config['num_classes'], kernel_size=(1, 1))
                model.classifier[1] = final_conv
            else:
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
        optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=config["sgd_momentum"])
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
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=1 / np.float32(config['lr_step']))
    else:
        raise ValueError(f'Unknown lr scheduler name {config["lr_scheduler"]}')
    return lr_scheduler
