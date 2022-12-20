"""
Training utility stuff
"""
import os
from glob import glob
import json
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset
import torch
from torchvision import transforms
from torchvision import models as tv_models
import pandas as pd
import numpy as np
import pydicom
import plotly
from clearml import Logger
import matplotlib.pyplot as plt
from my_dl_framework.evaluation.utils import validate_model_classification


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


def load_any_image(image_path: str) -> np.ndarray:
    """
    Loader for different images types, currently supports:
    *.dcm
    :param image_path:      Path to the image to load
    :return:                The image as numpy array
    """
    if image_path.endswith(".dcm"):
        dicom = pydicom.dcmread(image_path)
        image = dicom.pixel_array
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
        min_val, _ = torch.min(image.flatten(), dim=0)
        max_val, _ = torch.max(image.flatten(), dim=0)
        image = (image - min_val) / (max_val - min_val)
        return image


class RSNAChallengeBinaryDataset(Dataset):
    """
    Dataset for RSNA challenge classification task
    """
    def __init__(self, config: Dict, image_dir: str, subset: Union[List[str], None], is_training: bool, allow_missing_target: bool = False):
        self.config = config
        self.is_training = is_training
        self.subset = subset
        self.allow_missing_target = allow_missing_target
        # Load labels
        self.labels = pd.read_csv(os.path.join(self.config['base_path'], self.config['csv_name']))
        # Get images
        if self.subset is not None:
            self.image_paths = [file_name for file_name in glob(os.path.join(self.config['base_path'], image_dir, "*"))
                                if os.path.isfile(file_name) and
                                os.path.normpath(file_name).split(os.path.sep)[-1].split(".")[0] in self.subset]
        else:
            self.image_paths = [file_name for file_name in glob(os.path.join(self.config['base_path'], image_dir, "*"))
                                if os.path.isfile(file_name)]
        print("Len img paths", len(self.image_paths))
        self.images = list()
        if self.config['preload_images']:
            for image_path in self.image_paths:
                self.images.append(load_any_image(image_path))
        # Set up data augmentation
        transform_list = list()
        transform_list.append(transforms.ToTensor())
        transform_list.append(ZeroOneNorm())
        if self.config['resize_images'] is not None:
            transform_list.append(transforms.Resize(self.config['resize_images']))
        if self.is_training:
            if self.config['random_crop'] is not None:
                transform_list.append(transforms.RandomCrop(self.config['random_crop']))
            if self.config['random_fliplr'] is not None:
                transform_list.append(transforms.RandomHorizontalFlip())
            if self.config['color_jitter'] is not None:
                transform_list.append(transforms.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.2,
                    hue=0.2
                ))
        else:
            if self.config['apply_center_crop_inf'] is not None:
                transform_list.append(transforms.CenterCrop(self.config['random_crop']))
        self.all_transforms = transforms.Compose(transform_list)

    def __len__(self):
        """
        Length of the dataset
        :return:    length
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, int]:
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
        if base_file_name in self.labels['patientId']:
            label = int(self.labels.loc[self.labels['patientId'] == base_file_name].iloc[0]['Target'])
        elif self.allow_missing_target:
            label = 0
        else:
            raise ValueError(f"No label found for patient {base_file_name}")
        # Add channels
        image = np.concatenate((image, image, image), axis=0)
        # Data augmentation
        image = self.all_transforms(image)
        # TODO: take repeat label and idx for testtime aug
        return idx, image, label


def collate_aug_batch(batch):
    """ Collates an augmented batch, i.e., when the dataset returns
        returns multiple images at once.

    :param batch: Uncollated batch
    :return: collated batch
    """
    indices, imgs, targets = zip(*batch)
    return torch.cat(indices), torch.cat(imgs),torch.cat(targets)


def get_model(config: Dict) -> torch.nn.Module:
    """
    Get a torch model
    :param config:          Dict with config
    :return:                Torch model
    """
    if "TV_" in config["model_name"]:
        if config['model_name'] == "TV_densenet121":
            model = tv_models.densenet121(pretrained=config['pretrained'])
        if config['model_name'] == "TV_densenet161":
            model = tv_models.densenet161(pretrained=config['pretrained'])
        if config['model_name'] == "TV_densenet169":
            model = tv_models.densenet169(pretrained=config['pretrained'])
        if config['model_name'] == "TV_densenet201":
            model = tv_models.densenet201(pretrained=config['pretrained'])
        if config['model_name'] == "TV_efficientnet_v2_s":
            model = tv_models.efficientnet_v2_s(pretrained=config['pretrained'])
        if config['model_name'] == "TV_efficientnet_v2_m":
            model = tv_models.efficientnet_v2_m(pretrained=config['pretrained'])
        if config['model_name'] == "TV_efficientnet_v2_l":
            model = tv_models.efficientnet_v2_l(pretrained=config['pretrained'])
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
        Logger.current_logger().report_image(
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
