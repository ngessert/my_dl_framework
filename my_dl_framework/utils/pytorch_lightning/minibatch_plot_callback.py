from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
from typing import Dict, Any

from my_dl_framework.training.utils import plot_example_batch


class MBPlotCallback(Callback):
    def __init__(self, curr_subfolder_cv: str, config: Dict):
        self.curr_subfolder_cv = curr_subfolder_cv
        self.config = config
        self.state = {"has_plotted": False}

    @property
    def state_key(self):
        # note: we do not include `verbose` here on purpose
        return self._generate_state_key(config=self.config)

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        if not self.state["has_plotted"]:
            indices, images, targets = batch
            plot_example_batch(images.detach().cpu(), targets.detach().cpu(), batch_idx, self.curr_subfolder_cv, self.config)
            self.state["has_plotted"] = True

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)

    def state_dict(self):
        return self.state.copy()


