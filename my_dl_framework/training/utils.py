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


def load_any_image(image_path: str) -> np.ndarray:
    """
    Loader for different images types, currently supports:
    *.dcm
    :param image_path:      Path to the image to load
    :return:                The image as numpy array
    """
    if image_path.endswith(".dcm"):
        ds = pydicom.dcmread(image_path)
        image = ds.pixel_array
    else:
        raise ValueError(f'Unknown image type {image_path}')
    return image


class ZeroOneNorm:
    """
    0-1 norm for torch Tensors
    """
    def __init__(self):
        pass

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        min_val, _ = torch.min(image[...])
        max_val, _ = torch.max(image[...])
        image = (image - min_val) / (max_val - min_val)
        return image


class RSNAChallengeBinaryDataset(Dataset):
    """
    Dataset for RSNA challenge classification task
    """
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
        transform_list.append(ZeroOneNorm())
        if self.is_training:
            if self.config['resize_images'] is not None:
                transform_list.append(transforms.Resize(self.config['resize_images']))
            if self.config['random_crop'] is not None:
                transform_list.append(transforms.RandomCrop(self.config['random_crop']))
            if self.config['random_fliplr'] is not None:
                transform_list.append(transforms.RandomHorizontalFlip())
        self.all_transforms = transforms.Compose(transform_list)

    def __len__(self):
        """
        Length of the dataset
        :return:    length
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> (torch.Tensor, int):
        """
        Returns an image and label based on an index
        :param idx:         Index
        :return:
        """
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
    """
    Get a torch model
    :param config:          Dict with config
    :return:                Torch model
    """
    if config['model_name'] == "Densenet121":
        model = tv_models.densenet121(pretrained=config['pretrained'])
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, config['num_classes'])
    else:
        raise ValueError(f'Unknown model name {config["model_name"]}')
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


def save_optimizer_and_model(optimizer: torch.optim.Optimizer, model: torch.nn.Module, curr_subfolder:str, epoch: int, prefix: str):
    """
    Saves an optimizer and model
    :param optimizer:           Torch optimizer
    :param model:               Torch model
    :param curr_subfolder:      Subfolder to save in
    :param epoch:               Current epoch
    :param prefix:              Prefix in name
    :return: None
    """
    # Save model/optimizer
    state_opt = {'state_dict': optimizer.state_dict()}
    torch.save(state_opt, os.path.join(curr_subfolder, prefix + "_optimizer_ckpt_" + str(epoch) + ".pt"))
    state_model = {'state_dict': model.state_dict()}
    torch.save(state_model, os.path.join(curr_subfolder, prefix + "_model_ckpt_" + str(epoch) + ".pt"))
    # Remove previous one
    if os.path.exists(os.path.join(curr_subfolder, prefix + "_optimizer_ckpt_" + str(epoch - 1) + ".pt")):
        os.remove(os.path.join(curr_subfolder, prefix + "_optimizer_ckpt_" + str(epoch - 1) + ".pt"))
    if os.path.exists(os.path.join(curr_subfolder, prefix + "_model_ckpt_" + str(epoch - 1) + ".pt")):
        os.remove(os.path.join(curr_subfolder, prefix + "_model_ckpt_" + str(epoch - 1) + ".pt"))
