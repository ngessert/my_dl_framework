"""Ensemble script script. Works with or without targets. If there are targets, get metrics as well.

"""

import argparse
import shutil
from datetime import datetime
import yaml
import os
import numpy as np

from my_dl_framework.models.pl_class_wrapper import PLClassificationWrapper
import pytorch_lightning as pl
from clearml import Task, TaskTypes

from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML


def run_ensemble(config_path: str,
                 clearml: bool,
                 remote: bool):
    # Import config
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
        print(f'Using config {config_path}')

    # for storing ensemble output
    curr_ens_folder = os.path.join(os.path.normpath(config["save_path"]), config_path.split(os.sep)[-1] + "_" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    os.makedirs(curr_ens_folder, exist_ok=True)

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
        if "training_config" not in prev_task.artifacts:
            raise ValueError(f"ClearML task {clearml_id} does not training config file. Did you specify "
                             f"a successful evaluation task?")
        curr_folder = os.path.join(os.path.normpath(config['save_path']), clearml_id)
        os.makedirs(curr_folder, exist_ok=True)
        path_to_file = prev_task.artifacts["training_config"].get_local_copy()
        shutil.copy(path_to_file, os.path.join(curr_folder, "training_config.yaml"))
        if config["ensemble_type"] == "eval":
            if not os.path.exists(os.path.join(curr_folder, "predictions_all.npy")):
                path_to_file = prev_task.artifacts["pred arr _CV_cat"].get_local_copy()
                shutil.copy(path_to_file, os.path.join(curr_folder, "predictions_all.npy"))
            if not os.path.exists(os.path.join(curr_folder, "targets_all.npy")):
                path_to_file = prev_task.artifacts["tar arr _CV_cat"].get_local_copy()
                shutil.copy(path_to_file, os.path.join(curr_folder, "targets_all.npy"))
        elif config["ensemble_type"] == "predict":
            if not os.path.exists(os.path.join(curr_folder, "predictions_all.npy")):
                path_to_file = prev_task.artifacts["pred arr _CV_avg"].get_local_copy()
                shutil.copy(path_to_file, os.path.join(curr_folder, "predictions_all.npy"))
            if not os.path.exists(os.path.join(curr_folder, "targets_all.npy")):
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
    pred_list = list()
    tar_list = list()
    training_config = None
    for curr_subfolder in all_eval_folders:
        if not os.path.isfile(os.path.join(curr_subfolder, "predictions_all.npy")):
            raise ValueError(f"Could not find predictions_all.npy in folder {curr_subfolder}")
        if not os.path.isfile(os.path.join(curr_subfolder, "targets_all.npy")):
            raise ValueError(f"Could not find targets_all.npy in folder {curr_subfolder}")
        if training_config is None and os.path.isfile(os.path.join(curr_subfolder, "training_config.yaml")):
            with open(os.path.join(curr_subfolder, "training_config.yaml"), encoding="utf-8") as file:
                training_config = yaml.safe_load(file)
        pred = np.load(os.path.join(curr_subfolder, "predictions_all.npy"))
        pred_list.append(pred)
        tar_list.append(np.load(os.path.join(curr_subfolder, "targets_all.npy")))

    if training_config is None:
        raise ValueError("No training config file available - something seems to be wrong with your selected evaluation runs.")

    model = PLClassificationWrapper(config=training_config,
                                    training_set_len=0,
                                    prefix="dummy_model")
    # Aggregate CV predictions: concat for CV mode, average/vote for prediction mode
    pred_list = np.array(pred_list)
    tar_list = np.array(tar_list)
    if config["comb_type"] == "mean":
        pred_all = np.mean(pred_list, axis=0, keepdims=False)
        tar_all = np.mean(tar_list, axis=0, keepdims=False)
    else:
        raise ValueError(f"Unsupported ensemble combination type {config['comb_type']}")
    np.save(os.path.join(curr_ens_folder, "predictions_all.npy"), pred_all)
    np.save(os.path.join(curr_ens_folder, "targets_all.npy"), tar_all)
    if clearml:
        # Store intermediate preds from CV
        task.upload_artifact(f"pred arr {config['comb_type']}", os.path.join(curr_ens_folder, "predictions_all.npy"))
        task.upload_artifact(f"tar arr {config['comb_type']}", os.path.join(curr_ens_folder, "targets_all.npy"))
    # Perform evaluation through last model
    if config["ensemble_type"] == "eval":
        # we also want local logs
        csv_logger = pl.loggers.CSVLogger(curr_ens_folder, name="eval_local_logs")
        if task is not None:
            pl_clearml_logger = PLClearML(task=task, name_base="CV ensemble", title_prefix="_" + config['comb_type'])
        else:
            pl_clearml_logger = None
        loggers = [csv_logger]
        if pl_clearml_logger is not None:
            loggers.append(pl_clearml_logger)
        pl.seed_everything(42, workers=True)
        model.calc_and_log_metrics(loggers=loggers,
                                   predictions=pred_all,
                                   targets=tar_all,
                                   eval_name="ensemble",
                                   prefix="_" + config['comb_type'],
                                   step=0)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--config_path', type=str, help='Path to ensemble config', required=True)
    argparser.add_argument('-cl', '--clearml', action="store_true", help='Whether to use clearml for logging results', required=False)
    argparser.add_argument('-r', '--remote', action="store_true", help='Whether remote execution is done', required=False)

    args = argparser.parse_args()
    print(f'Args: {args}')
    run_ensemble(config_path=args.config_path,
                 clearml=args.clearml,
                 remote=args.remote)
