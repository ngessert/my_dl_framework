import torch
import numpy as np
import pydicom


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


def collate_aug_batch(batch):
    """ Collates an augmented batch, i.e., when the dataset returns
        returns multiple images at once.

    :param batch: Uncollated batch
    :return: collated batch
    """
    indices, imgs, targets = zip(*batch)
    return torch.cat(indices), torch.cat(imgs), torch.cat(targets)
