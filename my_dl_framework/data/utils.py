import torch
import numpy as np
import pydicom
from typing import List


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


class AddBatchDim:
    """
    Add batch dim
    """

    def __init__(self):
        pass

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.unsqueeze(0)


class FlipTTA:
    """
    Flip TTA augmentation. Expects a batch dim to be present
    """

    def __init__(self, flip_axis: int):
        self.flip_axis = flip_axis

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        result = torch.cat((image, torch.flip(image, dims=[self.flip_axis])))
        return result


class MultiEqualCrop:
    """
    Multi-crop evaluation with crops being equally distributed over the image. Expects a batch dim to be present
    """

    def __init__(self, num_per_axis: int, crop_size: int):
        self.num_per_axis = num_per_axis
        self.crop_size = crop_size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        crop_list = list()
        for i in range(self.num_per_axis):
            for j in range(self.num_per_axis):
                x_loc = int((self.crop_size / 2) + i * ((image.shape[2] - self.crop_size) / (self.num_per_axis - 1)))
                y_loc = int((self.crop_size / 2) + j * ((image.shape[2] - self.crop_size) / (self.num_per_axis - 1)))
                crop_img = image[:, :,
                                 x_loc - int(self.crop_size / 2): x_loc + int(self.crop_size / 2),
                                 y_loc - int(self.crop_size / 2): y_loc + int(self.crop_size / 2),
                                 ]
                crop_list.append(crop_img)
        result = torch.cat(crop_list)
        return result


def collate_aug_batch(batch):
    """ Collates an augmented batch, i.e., when the dataset returns
        returns multiple images at once.

    :param batch: Uncollated batch
    :return: collated batch
    """
    indices, imgs, targets = zip(*batch)
    return torch.cat(indices), torch.cat(imgs), torch.cat(targets)
