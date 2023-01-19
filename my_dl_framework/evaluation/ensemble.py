"""Ensemble script script. Works with or without targets. If there are targets, get metrics as well.

"""

import argparse
import ast
import shutil
from datetime import datetime
import json
import yaml
import os
import numpy as np
import torch
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader
from typing import Optional

from my_dl_framework.data.get_dataset import get_dataset
from my_dl_framework.data.utils import collate_aug_batch
from my_dl_framework.evaluation.utils import calculate_metrics
from my_dl_framework.models.pl_class_wrapper import PLClassificationWrapper
import pytorch_lightning as pl
from clearml import Task, TaskTypes

from my_dl_framework.utils.parse_str import parse_str
from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML
from my_dl_framework.utils.pytorch_lightning.minibatch_plot_callback import MBPlotCallback


def run_prediction(config_path: str,
                   clearml: bool,
                   remote: bool,
                   ):
    # Import config
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
        print(f'Using config {config_path}')

    all_eval_folders = config["eval_folders"] or list()
    for clearml_id in config['clearml_ids']:
        print(f"Retrieving artifacts from ClearML ID {clearml_id}")
        prev_task = Task.get_task(task_id=clearml_id)
        # Check if required artifact exists
        if "pred arr _CV_avg" not in prev_task.artifacts and config["ensemble_type"] == "predict":
            raise ValueError(f"ClearML task {clearml_id} does not have a prediction-style predictions file. Did you "
                             f"specify a successful evaluation task?")
        if "tar arr _CV_avg" not in prev_task.artifacts and config["ensemble_type"] == "predict":
            raise ValueError(f"ClearML task {clearml_id} does not have a prediction-style targets file. Did you "
                             f"specify a successful evaluation task?")
        if "pred arr _CV_cat" not in prev_task.artifacts and config["ensemble_type"] == "eval":
            raise ValueError(f"ClearML task {clearml_id} does not have a CV-eval-style predictions file. Did you "
                             f"specify a successful evaluation task?")
        if "tar arr _CV_cat" not in prev_task.artifacts and config["ensemble_type"] == "eval":
            raise ValueError(f"ClearML task {clearml_id} does not have a CV-eval-style targets file. Did you specify "
                             f"a successful evaluation task?")
        curr_folder = os.path.join(os.path.normpath(config['save_path']), clearml_id)
        os.makedirs(curr_folder, exist_ok=True)
        if config["ensemble_type"] == "eval":
            path_to_file = prev_task.artifacts["pred arr _CV_cat"].get_local_copy()
            shutil.copy(path_to_file, os.path.join(curr_folder, "predicitions_all.npy"))
            path_to_file = prev_task.artifacts["tar arr _CV_cat"].get_local_copy()
            shutil.copy(path_to_file, os.path.join(curr_folder, "targets_all.npy"))
        elif config["ensemble_type"] == "predict":
            path_to_file = prev_task.artifacts["pred arr _CV_avg"].get_local_copy()
            shutil.copy(path_to_file, os.path.join(curr_folder, "predicitions_all.npy"))
            path_to_file = prev_task.artifacts["tar arr _CV_avg"].get_local_copy()
            shutil.copy(path_to_file, os.path.join(curr_folder, "targets_all.npy"))
        # Add to normal folder list
        all_eval_folders.append(curr_folder)
    # ClearML
    task = None
    if clearml:
        if remote:
            task = Task.get_task(project_name="RSNABinary", task_name=config_path.split(os.sep)[-1])  # ?
        else:
            task = Task.init(project_name='RSNABinary',
                             task_name=config_path.split(os.sep)[-1],
                             reuse_last_task_id=False,
                             auto_connect_frameworks={'matplotlib': False},
                             task_type=TaskTypes.inference
                             )
        task.connect(config)
        task.add_tags(config["clearml_tags"])
    # for storing ensemble output
    curr_ens_folder = os.path.join(os.path.normpath(config["save_path"]), config_path.split(os.sep)[-1] + "_" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    os.makedirs(curr_ens_folder, exist_ok=True)
    pred_list = list()
    tar_list = list()
    loggers = list()
    for curr_subfolder in glob(os.path.join(all_eval_folders, "/*")):
        if not os.path.isfile(os.path.join(curr_subfolder, "predictions_all.npy")):
            raise ValueError(f"Could not find predictions_all.npy in folder {curr_subfolder}")
        if not os.path.isfile(os.path.join(curr_subfolder, "targets_all.npy")):
            raise ValueError(f"Could not find targets_all.npy in folder {curr_subfolder}")
        # we also want local logs
        csv_logger = pl.loggers.CSVLogger(curr_ens_folder, name="eval_local_logs")
        if task is not None:
            pl_clearml_logger = PLClearML(task=task, name_base="CV" + str(cv_idx+1), title_prefix=CV_PREFIX)
        else:
            pl_clearml_logger = None
        loggers = [csv_logger]
        if pl_clearml_logger is not None:
            loggers.append(pl_clearml_logger)
        pl.seed_everything(42, workers=True)

    model = PLClassificationWrapper(config=config,
                                    training_set_len=0,
                                    prefix="dummy_model")
    # Aggregate CV predictions: concat for CV mode, average/vote for prediction mode
    pred_list = np.array(pred_list)
    tar_list = np.array(tar_list)
    if config["ensemble_type"] == "eval":
        prefix = "_CV_cat"
        pred_all = np.concatenate(pred_list, axis=0).reshape([-1, pred_list[0].shape[-1]])
        tar_all = np.concatenate(tar_list, axis=0).reshape([-1])
    else:
        prefix = "_CV_avg"
        pred_all = np.squeeze(np.mean(pred_list, axis=0, keepdims=False), axis=0)
        tar_all = np.squeeze(np.mean(tar_list, axis=0, keepdims=False), axis=0)
    np.save(os.path.join(curr_ens_folder, "predictions_all.npy"), pred_all)
    np.save(os.path.join(curr_ens_folder, "targets_all.npy"), tar_all)
    if clearml:
        # Store intermediate preds from CV
        task.upload_artifact(f"pred arr {prefix}", pred_all)
        task.upload_artifact(f"tar arr {prefix}", tar_all)
    # Perform evaluation through last model
    if config["ensemble_type"] == "eval":
        model.calc_and_log_metrics(loggers=loggers,
                                   predictions=pred_all,
                                   targets=tar_all,
                                   eval_name="ensemble",
                                   prefix=prefix,
                                   step=0)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--folder', type=str, help='Folder path', required=False)
    argparser.add_argument('-cid', '--clearml_id', type=str, help='ClearML ID of experiment', required=False)
    argparser.add_argument('-cl', '--clearml', action="store_true", help='Whether to use clearml for logging results', required=False)
    argparser.add_argument('-r', '--remote', action="store_true", help='Whether remote execution is done', required=False)
    argparser.add_argument('-dls', '--dont_log_splits', action="store_true", help='Whether inidivudal CV splits should be logged or just the cat/avg', required=False)
    argparser.add_argument('-etr', '--eval_on_train', action="store_true", help='Whether to evaluate on the training set (when doing CV evaluation)', required=False)
    argparser.add_argument('-cv', '--cv_eval', action="store_true", help='Whether the CV split file should be used for evaluation', required=False)
    argparser.add_argument('-t', '--image_path', type=str, default=None, help='Test image path for prediction', required=False)
    argparser.add_argument('-lcp', '--label_csv_path', type=str, default=None, help='Optional path to a label CSV that overrides the default one (CV eval) or is used for predict mode', required=False)
    argparser.add_argument('-l', '--limit_batches', type=int, default=None, help='Test with a subset of batches', required=False)

    argparser.add_argument('-ttafh', '--tta_flip_horz', action="store_true", help='Test-time augmentation: horizontal flip', required=False)
    argparser.add_argument('-ttafv', '--tta_flip_vert', action="store_true", help='Test-time augmentation: vertical flip', required=False)
    argparser.add_argument('-ttameq', '--tta_multi_eq_crop', type=int, default=0, help='Test-time augmentation: multiple equal crops', required=False)
    args = argparser.parse_args()
    print(f'Args: {args}')
    run_prediction(folder=args.folder,
                   clearml_id=args.clearml_id,
                   clearml=args.clearml,
                   remote=args.remote,
                   eval_on_train=args.eval_on_train,
                   cv_eval=args.cv_eval,
                   image_path=args.image_path,
                   limit_batches=args.limit_batches,
                   dont_log_splits=args.dont_log_splits,
                   label_csv_path=args.label_csv_path,
                   tta_flip_horz=args.tta_flip_horz,
                   tta_flip_vert=args.tta_flip_vert,
                   tta_multi_eq_crop=args.tta_multi_eq_crop)
