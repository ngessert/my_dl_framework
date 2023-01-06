import os
from glob import glob
from typing import Dict, List, Tuple, Union
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import pandas as pd
import numpy as np

from my_dl_framework.data.utils import load_any_image, ZeroOneNorm


class RSNAChallengeBinaryDataset(Dataset):
    """
    Dataset for RSNA challenge classification task
    """
    def __init__(self, config: Dict, image_dir: str, path_to_label_csv: Union[str, None], subset: Union[List[str], None], is_training: bool,
                 allow_missing_target: bool = False):
        self.config = config
        self.is_training = is_training
        self.subset = subset
        self.allow_missing_target = allow_missing_target
        # Load labels
        if path_to_label_csv is not None:
            self.labels = pd.read_csv(path_to_label_csv)
        else:
            self.labels = None
        # Get images
        if self.subset is not None:
            self.image_paths = [file_name for file_name in glob(os.path.join(image_dir, "*"))
                                if os.path.isfile(file_name) and
                                os.path.normpath(file_name).split(os.path.sep)[-1].split(".")[0] in self.subset]
        else:
            self.image_paths = [file_name for file_name in glob(os.path.join(image_dir, "*"))
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
        if self.labels is None:
            label = 0
        elif (self.labels['patientId'].eq(base_file_name)).any():
            label = int(self.labels.loc[self.labels['patientId'] == base_file_name].iloc[0]['Target'])
        elif self.allow_missing_target:
            label = 0
        else:
            raise ValueError(f"No label found for patient {base_file_name}")
        # Add channels
        image = np.concatenate((image[:, :, None], image[:, :, None], image[:, :, None]), axis=2)
        # Data augmentation
        image = self.all_transforms(image)
        # TODO: take repeat label and idx for testtime aug
        return idx, image, label
