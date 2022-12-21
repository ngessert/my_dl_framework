from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import pytorch_lightning as pl

from my_dl_framework.evaluation.utils import calculate_metrics
from my_dl_framework.training.utils import get_lossfunction, get_lr_scheduler, get_tv_class_model, get_optimizer
from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML


def stack_list_of_dicts(list_of_dicts: Dict[str, Any], keys: List[str], to_numpy: bool = False) -> Dict[str, np.ndarray]:
    """
    Takes a list of dicts and concatenates the tensors in them, then converts them to numpy
    :param list_of_dicts:       List of dicts with tensors as values
    :param keys:                Subset of keys from the dict to use
    :param to_numpy:            Convert to numpy yes/no
    :return:                    New dict with stacked tensors, as ndarrays
    """
    stacked_dict = dict()
    for key in keys:
        stacked_dict[key] = list_of_dicts[0][key]
    for i, out_dict in enumerate(list_of_dicts):
        # First one is covered
        if i == 0:
            continue
        for key in keys:
            stacked_dict[key] = torch.cat((stacked_dict[key], out_dict[key]))
    if to_numpy:
        for key in keys:
            stacked_dict[key] = stacked_dict[key].detach().cpu().numpy()
    return stacked_dict


class PLClassificationWrapper(pl.LightningModule):
    """ Pytorch lightning wrapper for classification models
    """
    def __init__(self, config: Dict, 
                 training_set_len: int = 0):
        super().__init__()
        self.config = config
        self.model = get_tv_class_model(config=self.config)
        self.loss_function = get_lossfunction(config=self.config)
        self.training_set_len = training_set_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.model(x)
        return res

    def configure_optimizers(self):
        optimizer = get_optimizer(self.config, self.model)
        lr_scheduler = get_lr_scheduler(self.config, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, train_batch, batch_idx):
        _, images, targets = train_batch
        outputs = self.model(images)
        loss = self.loss_function(outputs, targets)
        return {"loss": loss, "batch_idx": batch_idx}

    def training_step_end(self, step_output):
        batch_idx = step_output["batch_idx"][0]
        loss = torch.stack(step_output["loss"])
        # Track loss
        if batch_idx % self.config["loss_log_freq"] == 0:
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"loss": loss, "batch_idx": batch_idx}

    def validation_step(self, val_batch, batch_idx, **kwargs):
        indices, images, targets = val_batch
        outputs = F.softmax(self.model(images), dim=1)
        return_dict = {"outputs": outputs,
                       "targets": targets,
                       "indices": indices}
        if "dataloader_idx" in kwargs:
            return_dict["dataloader_idx"] = kwargs["dataloader_idx"]
        else:
            return_dict["dataloader_idx"] = 0
        return return_dict

    def validation_step_end(self, step_output):
        if isinstance(step_output, list):
            stacked_dict = stack_list_of_dicts(step_output, ["outputs", "targets", "indices"], to_numpy=False)
        else:
            stacked_dict = step_output
        return {"outputs": stacked_dict["outputs"],
                "targets": stacked_dict["targets"],
                "indices": stacked_dict["indices"],
                "dataloader_idx": step_output["dataloader_idx"]}

    def validation_epoch_end(self, outputs):
        # Assumes val loader is put first
        eval_name = "validation" if outputs[0]["dataloader_idx"] == 0 else "train_val"
        stacked_dict = stack_list_of_dicts(outputs, ["outputs", "targets", "indices"], to_numpy=True)
        stacked_preds = stacked_dict["outputs"]
        stacked_targets = stacked_dict["targets"]
        stacked_indices = stacked_dict["indices"]
        predictions = list()
        targets = list()
        for idx in np.unique(stacked_indices):
            if self.config["test_aug_ensemble_mode"] == "mean":
                predictions.append(np.mean(stacked_preds[stacked_indices == idx], 0))
                targets.append(np.mean(stacked_targets[stacked_indices == idx]))
            else:
                predictions.append(stacked_preds[stacked_indices == idx][0, :])
                targets.append(stacked_targets[stacked_indices == idx][0])
        predictions = np.array(predictions)
        targets = np.array(targets)
        metrics, plots = calculate_metrics(predictions=predictions,
                                           targets=targets,
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
