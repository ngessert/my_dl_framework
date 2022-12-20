from typing import Dict
import pandas as pd
import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from my_dl_framework.evaluation.utils import calculate_metrics
from my_dl_framework.training.utils import get_lossfunction, get_lr_scheduler, get_model, get_optimizer
from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML


class PLClassificationWrapper(pl.LightningModule):
    """ Pytorch lightning wrapper for classification models
    """
    def __init__(self, config: Dict, 
                 training_set_len: int = 0):
        super().__init__()
        self.config = config
        self.modules = get_model(config=self.config)
        self.loss_function = get_lossfunction(config=self.config)
        self.training_set_len = training_set_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.modules(x)
        return res

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.modules)
        lr_scheduler = get_lr_scheduler(self.config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, train_batch, batch_idx):
        images, targets = train_batch
        outputs = self.model(images)
        loss = self.loss_function(outputs, targets)
        return {"loss": loss, "batch_idx": batch_idx}

    def training_step_end(self, step_output):
        batch_idx = step_output["batch_idx"][0]
        loss = torch.stack(step_output["loss"])
        # Track loss
        if batch_idx % self.config["loss_log_freq"] == 0:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return 

    def validation_step(self, val_batch, batch_idx, dataloader_idx):
        images, targets = val_batch
        outputs = F.softmax(self.modules(images), dim=1)
        return {"outputs": outputs, "targets": targets, "dataloader_idx": dataloader_idx}

    def validation_step_end(self, step_output):
        return {"outputs": torch.stack(step_output["outputs"]),
                "targets": torch.stack(step_output["targets"]),
                "dataloader_idx": step_output["dataloader_idx"][0]}

    def validation_epoch_end(self, outputs):
        # Assumes val loader is put first
        eval_name = "validation" if outputs["dataloader_idx"][0] == 0 else "train_val"
        outputs = torch.stack(outputs["outputs"])
        targets = torch.stack(outputs["targets"])
        metrics, plots = calculate_metrics(predictions=outputs.deatch().numpy(),
                                           targets=targets.detach().numpy(),
                                           class_names=self.config["class_names"])
        for i in range(len(self.config["class_names"])):
            self.log_dict({key + "_" + self.config["class_names"][i]: val[i] for key, val in metrics.items()})
        self.log_dict({key + "_mean": val[-1] for key, val in metrics.items()})
        for logger in self.loggers:
            if isinstance(logger, PLClearML):
                logger.log_plotly(plotly_obj_dict=plots,
                                  step=self.current_epoch)
                metric_df = pd.DataFrame.from_dict(metrics)
                metric_df["classes"] = self.config["class_names"] + ["Avg"]
                metric_df.set_index("classes", inplace=True)
                logger.log_table(title="Metrics " + eval_name,
                                 step=self.current_epoch,
                                 dataframe=metric_df)
