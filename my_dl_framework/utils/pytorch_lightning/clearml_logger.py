"""
ClearML Logger
-------------------------
"""
import os
from argparse import Namespace
from pathlib import Path
import shutil
from typing import Any, Dict, Mapping, Optional, Union
import pandas as pd

from torch import Tensor

from lightning_lite.utilities.types import _PATH
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities.logger import (
    _convert_params,
    _flatten_dict,
    _sanitize_callable_params,
    _scan_checkpoints,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from clearml import Task


class PLClearML(Logger):
    """ ClearML logger for pytorch lightning
    """

    LOGGER_JOIN_CHAR = "-"

    def __init__(
            self,
            task: Task,
            name_base: str,
            save_dir: _PATH = ".",
            title_prefix: str = "",
            log_model: bool = True
    ) -> None:
        super().__init__()
        self.task = task
        self.name_base = name_base
        self.title_prefix = title_prefix
        self.logger = task.get_logger()
        self._save_dir = save_dir
        self._log_model = log_model

        self._logged_model_time = dict()
        self._checkpoint_callback = None

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        return state

    @property
    def name(self) -> str:
        return self.name_base + "_" + self.task.name

    @property
    def version(self):
        return "version_0"

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _flatten_dict(params)
        params = _sanitize_callable_params(params)
        self.task.connect_configuration(params)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

        if step is None:
            step = 0
        for metric in metrics:
            self.logger.report_scalar(title=metric.split(self.title_prefix)[0],
                                      series=metric,
                                      value=metrics[metric],
                                      iteration=step)

    @rank_zero_only
    def log_table(
            self,
            title: str,
            dataframe: pd.DataFrame = None,
            step: Optional[int] = None,
    ) -> None:
        """Log a pandas table
        """
        if step is None:
            step = 0
        self.logger.report_table(title=title,
                                 series=title,
                                 iteration=step,
                                 table_plot=dataframe)

    @rank_zero_only
    def log_plotly(
            self,
            title: str,
            plotly_obj_dict: Dict[str, Any],
            step: Optional[int] = None,
    ) -> None:
        """ Log list of plotly objects to clearml.
        """
        if step is None:
            step = 0
        for plot_name, plot in plotly_obj_dict.items():
            self.logger.report_plotly(title=plot_name + "_" + title,
                                      series=plot_name + "_" + title,
                                      figure=plot,
                                      iteration=step)

    @rank_zero_only
    def log_text(
            self,
            message: str,
    ) -> None:
        """Log text in clearml.
        """
        self.logger.report_text(msg=message)

    @rank_zero_only
    def log_image(self, title: str, image, step: Optional[int] = None) -> None:
        """Log one image.

        """
        if step is None:
            step = 0
        self.logger.report_image(title=title.split(self.title_prefix)[0],
                                 series=title,
                                 image=image,
                                 iteration=step)

    @property
    def save_dir(self) -> Optional[str]:
        """Gets the save directory.

        Returns:
            The path to the save directory.
        """
        return self._save_dir

    def after_save_checkpoint(self, checkpoint_callback: ModelCheckpoint) -> None:
        # log checkpoints as artifacts
        if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
            self._scan_and_log_checkpoints(checkpoint_callback)
        elif self._log_model is True:
            self._checkpoint_callback = checkpoint_callback

    @rank_zero_only
    def download_artifact(
            self,
            artifact: str,
            save_dir: Optional[_PATH] = None,
    ) -> str:
        """Downloads an artifact from the clearml server.

        Args:
            artifact: The path of the artifact to download.
            save_dir: The directory to save the artifact to.

        Returns:
            The path to the downloaded artifact.
        """
        tmp_path = self.task.artifacts[artifact].get_local_copy()
        shutil.move(tmp_path, save_dir)
        return os.path.join(save_dir, tmp_path.split(os.sep)[-1])

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if status != "success":
            # Currently, checkpoints only get logged on success
            return
        # log checkpoints as artifacts
        if self._checkpoint_callback:
            self._scan_and_log_checkpoints(self._checkpoint_callback)

    def _scan_and_log_checkpoints(self, checkpoint_callback: ModelCheckpoint) -> None:
        # get checkpoints to be saved with associated score
        checkpoints = _scan_checkpoints(checkpoint_callback, self._logged_model_time)

        # log iteratively all new checkpoints
        for ckpt_time, ckpt_path, ckpt_score, tag in checkpoints:
            metadata = (
                {
                    "score": ckpt_score.item() if isinstance(ckpt_score, Tensor) else ckpt_score,
                    "original_filename": Path(ckpt_path).name,
                    "tag": tag,
                    checkpoint_callback.__class__.__name__: {
                        k: getattr(checkpoint_callback, k)
                        for k in [
                            "monitor",
                            "mode",
                            "save_last",
                            "save_top_k",
                            "save_weights_only",
                            "_every_n_train_steps",
                        ]
                        # ensure it does not break if `ModelCheckpoint` args change
                        if hasattr(checkpoint_callback, k)
                    },
                }
            )
            self.task.upload_artifact(name=f"model-{self.name_base}-{tag}",
                                      artifact_object=ckpt_path,
                                      metadata=metadata)
            # remember logged models - timestamp needed in case filename didn't change (lastkckpt or custom name)
            self._logged_model_time[ckpt_path] = ckpt_time
