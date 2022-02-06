"""
Training utility stuff
"""
from typing import Dict, List
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision import models as tv_models
import pandas as pd
import os
from glob import glob
import numpy as np
import pydicom


def get_dataset(config: Dict, image_dir: str, subset: List[str], is_training: bool) -> Dataset:
    if config["dataset_type"] == "RSNAChallengeBinary":
        return RSNAChallengeBinaryDataset(config=config, image_dir=image_dir, subset=subset, is_training=is_training)
    else:
        raise ValueError(f'Unknown dataset type {config["dataset_type"]}')


def load_any_image(image_path: str) -> np.ndarray:
    if image_path.endswith(".dcm"):
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array
    else:
        raise ValueError(f'Unknown image type {image_path}')
    return image


class RSNAChallengeBinaryDataset(Dataset):
    def __init__(self, config: Dict, image_dir: str, subset: List[str], is_training: bool):
        self.config = config
        self.is_training = is_training
        self.subset = subset
        # Load labels
        self.labels = pd.read_csv(os.path.join(self.config['base_path'], self.config['csv_name']))
        # Get images
        self.image_paths = [file_name for file_name in glob(os.path.join(self.config['base_path'], image_dir, "*"))
                            if os.path.isfile(file_name) and
                            os.path.normpath(file_name).split(os.path.sep)[-1].split(".")[0] in self.subset]
        print("Len img paths", len(self.image_paths))
        self.images = list()
        if self.config['preload_images']:
            for image_path in self.image_paths:
                self.images.append(load_any_image(image_path))
        # Set up data augmentation
        transform_list = list()
        transform_list.append(transforms.ToTensor())
        if self.is_training:
            if self.config['resize_images'] is not None:
                transform_list.append(transforms.Resize(self.config['resize_images']))
            if self.config['random_crop'] is not None:
                transform_list.append(transforms.RandomCrop(self.config['random_crop']))
            if self.config['random_fliplr'] is not None:
                transform_list.append(transforms.RandomHorizontalFlip())
        self.all_transforms = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.config['preload_images']:
            image = self.images[idx]
        else:
            image = load_any_image(self.image_paths[idx])
        # Label from csv
        base_file_name = os.path.normpath(self.image_paths[idx]).split(os.path.sep)[-1].split(".")[0]
        label = int(self.labels.loc[self.labels['patientId'] == base_file_name].iloc[0]['Target'])
        # Data augmentation
        image = self.all_transforms(image)
        # Add channels
        image = torch.cat((image, image, image), dim=0)
        return image, label


def get_model(config: Dict) -> torch.nn.Module:
    if config['model_name'] == "Densenet121":
        model = tv_models.densenet121(pretrained=config['pretrained'])
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, config['num_classes'])
    else:
        raise ValueError(f'Unknown model name {config["model_name"]}')
    return model


def get_lossfunction(config: Dict) -> torch.nn.Module:
    if config['loss_name'] == "cross_entropy":
        loss_function = torch.nn.CrossEntropyLoss(weight=config['loss_weights'] if 'loss_weights' in config else None)
    else:
        raise ValueError(f'Unknown loss name {config["loss_name"]}')
    return loss_function


def get_optimizer(config: Dict, model: torch.nn.Module) -> torch.optim.Optimizer:
    if config['optimizer'] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        raise ValueError(f'Unknown optimizer name {config["optimizer"]}')
    return optimizer


def get_lr_scheduler(config: Dict, optimizer: torch.optim.Optimizer):
    if config['lr_scheduler'] == "stepwise":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=1/np.float32(config['lr_step']))
    else:
        raise ValueError(f'Unknown lr scheduler name {config["lr_scheduler"]}')
    return lr_scheduler
