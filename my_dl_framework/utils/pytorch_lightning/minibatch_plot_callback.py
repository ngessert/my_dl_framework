import os
from typing import (Any, Dict)

import pytorch_lightning as pl
import torch
from clearml import Logger
from lightning.pytorch.callbacks import Callback
from matplotlib import pyplot as plt


def plot_example_batch(images: torch.Tensor, targets: torch.Tensor, idx: int, save_path: str, config: Dict):
    """ Plots some example batches into clearml and to local path
    """
    batch_size = images.shape[0]
    fig, axes = plt.subplots(batch_size, 1, figsize=(5 * batch_size, 20))
    for i in range(batch_size):
        image = images[i, 0, :, :].numpy()
        target = config['data']['class_names'][targets[i].item()]
        clearml_logger = Logger.current_logger()
        if clearml_logger is not None:
            clearml_logger.report_image(
                f"{i} of {idx} y={target}",
                'debug example',
                iteration=0,
                image=image,
            )
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f'Target {target}', fontsize=5)
        axes[i].set_axis_off()
    os.makedirs(os.path.join(save_path, 'example_batches'), exist_ok=True)
    fig.savefig(os.path.join(save_path, 'example_batches', 'batch_' + str(idx) + '.png'), bbox_inches='tight', pad_inches=0, dpi=300)


class MBPlotCallback(Callback):

    def __init__(self, curr_subfolder_cv: str, config: Dict):
        self.curr_subfolder_cv = curr_subfolder_cv
        self.config = config
        self.state = {'has_plotted': False}

    @property
    def state_key(self):
        # note: we do not include `verbose` here on purpose
        return self._generate_state_key(config=self.config)

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        if not self.state['has_plotted']:
            _, images, targets = batch
            plot_example_batch(images.detach().cpu(), targets.detach().cpu(), batch_idx, self.curr_subfolder_cv, self.config)
            self.state['has_plotted'] = True

    def on_validation_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        if not self.state['has_plotted']:
            _, images, targets = batch
            plot_example_batch(images.detach().cpu(), targets.detach().cpu(), batch_idx, self.curr_subfolder_cv, self.config)
            self.state['has_plotted'] = True

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()
