"""Prediction script. Works with or without targets. If there are targets, get metrics as well.

"""

import argparse
import shutil
from datetime import datetime
import json
import yaml
import os
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from typing import Optional

from my_dl_framework.data.get_dataset import get_dataset
from my_dl_framework.data.utils import collate_aug_batch
from my_dl_framework.models.pl_class_wrapper import PLClassificationWrapper
import pytorch_lightning as pl
from clearml import Task, TaskTypes

from my_dl_framework.utils.parse_str import parse_str
from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML
from my_dl_framework.utils.pytorch_lightning.minibatch_plot_callback import MBPlotCallback


def run_prediction(folder: str,
                   clearml_id: Optional[str],
                   clearml: bool,
                   remote: bool,
                   eval_on_train: bool,
                   cv_eval: bool,
                   image_path: str,
                   limit_batches: int,
                   dont_log_splits: bool,
                   label_csv_path: str,
                   tta_flip_horz: bool,
                   tta_flip_vert: bool,
                   tta_multi_eq_crop: int):
    # Constants
    CV_PREFIX = "_CV_"

    # TTA options
    batch_size_factor = 1
    tta_options = dict()
    if tta_flip_horz:
        tta_options["flip_horz"] = True
        batch_size_factor *= 2
    if tta_flip_vert:
        tta_options["flip_vert"] = True
        batch_size_factor *= 2
    if tta_multi_eq_crop > 0:
        tta_options["multi_eq_crop"] = tta_multi_eq_crop
        batch_size_factor *= tta_multi_eq_crop

    image_path = os.path.normpath(image_path)
    if clearml_id is not None:
        print(f"Retrieving artifacts from ClearML ID {clearml_id}")
        prev_task = Task.get_task(task_id=clearml_id)
        os.makedirs(folder, exist_ok=True)
        # Checkpoints per CV file
        cv_idx = 1
        fold_exists = True
        while fold_exists:
            artifact_str = f"model-CV{cv_idx}-latest"
            if os.path.isfile(os.path.join(folder, f"CV_{cv_idx}", "last.ckpt")):
                cv_idx += 1
                continue
            if artifact_str in prev_task.artifacts:
                print(f"Found ckpt for fold {cv_idx}, downloading...")
                path_to_file = prev_task.artifacts[artifact_str].get_local_copy()
                os.makedirs(os.path.join(folder, f"CV_{cv_idx}"), exist_ok=True)
                shutil.copy(path_to_file, os.path.join(folder, f"CV_{cv_idx}", "last.ckpt"))
                cv_idx += 1
            else:
                fold_exists = False
        # Config
        config_dict = prev_task.get_parameters_as_dict()
        config_dict = {key: parse_str(val) for key, val in config_dict["General"].items()}
        with open(os.path.join(folder, "config.yaml"), "w", encoding="utf-8") as file:
            yaml.safe_dump(config_dict, file)
        # CV split file
        artifact_str = f"cv_split_file"
        if artifact_str in prev_task.artifacts:
            print(f"CV split file found, downloading...")
            path_to_file = prev_task.artifacts[artifact_str].get_local_copy()
            shutil.copy(path_to_file, os.path.join(folder, "cv_split_file.json"))
    elif folder is None or not os.path.exists(folder):
        raise ValueError(f"Provided subfolder does not exist. ({folder})")
    if not os.path.isfile(os.path.join(folder, "config.yaml")):
        raise ValueError(f"No config.yaml found in folder {folder}.")
    # Import config
    with open(os.path.join(folder, "config.yaml"), encoding="utf-8") as file:
        config = yaml.safe_load(file)
        print(f'Using config {os.path.join(folder, "config.yaml")}')
    data_split = None
    if cv_eval:
        cv_split_file = os.path.join(folder, "cv_split_file.json")
        if not os.path.isfile(cv_split_file):
            raise ValueError("Could not find a CV in the folder although CV eval was executed")
        with open(cv_split_file, encoding="utf-8") as file:
            data_split = json.load(file)
            print(f'Using CV split file {cv_split_file}')
    # ClearML
    task = None
    if clearml:
        if remote:
            task = Task.get_task(project_name="RSNABinary", task_name=folder.split(os.sep)[-1])  # ?
        else:
            task = Task.init(project_name='RSNABinary',
                             task_name=folder.split(os.sep)[-1],
                             reuse_last_task_id=False,
                             auto_connect_frameworks={'matplotlib': False},
                             task_type=TaskTypes.inference
                             )
        task.connect({"folder": folder, "clearml_id": clearml_id, "eval_on_train": eval_on_train,
                      "cv_eval": cv_eval, "image_path": image_path, "limit_batches": limit_batches,
                      "dont_log_splits": dont_log_splits})
        tags = list()
        if cv_eval:
            tags.append("CVEval")
        else:
            tags.append("Predict")
        if eval_on_train:
            tags.append("EvalOnTrain")
        if limit_batches is not None:
            tags.append(f"Lim{limit_batches}")
        if dont_log_splits:
            tags.append("NoSplits")
        if tta_flip_horz:
            tags.append("TTAFlipH")
        if tta_flip_vert:
            tags.append("TTAFlipV")
        if tta_multi_eq_crop:
            tags.append(f"TTAMEQ{tta_multi_eq_crop}")
        task.add_tags(tags)
    # for storing predictions, there may be many evaluations
    curr_pred_folder = os.path.join(folder, "pred_and_eval_" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    os.makedirs(curr_pred_folder, exist_ok=True)
    pred_list = list()
    pred_train_val_list = list()
    tar_list = list()
    tar_train_val_list = list()
    model = None
    loggers = list()
    for curr_subfolder_cv in glob(os.path.join(folder, "CV_*")):
        if not os.path.isdir(curr_subfolder_cv):
            continue
        cv_idx = int(curr_subfolder_cv.split("CV_")[-1]) - 1
        if not os.path.exists(os.path.join(curr_subfolder_cv, "last.ckpt")):
            print(f"No checkpoint found for fold {cv_idx+1}, continue...")
            continue
        print(f'Starting fold number {cv_idx+1}')
        if data_split is not None:
            subset_train = data_split["training_splits"][cv_idx]
            subset_val = data_split["validation_splits"][cv_idx]
        else:
            subset_train = list()
            subset_val = [file_path.split(os.sep)[-1].split(".")[0] for file_path in glob(image_path + "/*")]
        # Data
        if label_csv_path is not None:
            path_to_label_csv = label_csv_path
        elif cv_eval:
            path_to_label_csv = os.path.join(image_path.replace(image_path.split(os.sep)[-1], ""),
                                                                 config['csv_name'])
        else:
            path_to_label_csv = None
        dataset_val = get_dataset(config=config,
                                  image_dir=image_path,
                                  path_to_label_csv=path_to_label_csv,
                                  subset=subset_val,
                                  tta_options=tta_options,
                                  is_training=False)
        print(f'Size validation dataset {len(dataset_val)}')
        batcH_size_adj = config["batch_size"] // batch_size_factor
        if batcH_size_adj <= 0:
            batcH_size_adj = 1
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=batcH_size_adj, shuffle=False,
                                    num_workers=8, pin_memory=True, collate_fn=collate_aug_batch if tta_options else None)
        if eval_on_train and cv_eval:
            dataset_train_val = get_dataset(config=config,
                                            image_dir=image_path,
                                            path_to_label_csv=path_to_label_csv,
                                            subset=subset_train,
                                            tta_options=tta_options,
                                            is_training=False)
            print(f'Size validation dataset {len(dataset_train_val)}')
            dataloader_train_val = DataLoader(dataset=dataset_train_val, batch_size=batcH_size_adj, shuffle=False,
                                              num_workers=8, pin_memory=True, collate_fn=collate_aug_batch if tta_options else None)
        else:
            dataloader_train_val = None
        model = PLClassificationWrapper(config=config,
                                        training_set_len=len(dataset_val),
                                        prefix=CV_PREFIX + str(cv_idx+1))
        # grab from model and save
        curr_subfolder_cv_pred = os.path.join(curr_pred_folder, "CV_" + str(cv_idx + 1))
        # we also want local logs
        csv_logger = pl.loggers.CSVLogger(curr_subfolder_cv_pred, name="eval_local_logs")
        if task is not None:
            pl_clearml_logger = PLClearML(task=task, name_base="CV" + str(cv_idx+1), title_prefix=CV_PREFIX)
        else:
            pl_clearml_logger = None
        loggers = [csv_logger]
        if pl_clearml_logger is not None:
            loggers.append(pl_clearml_logger)
        if dont_log_splits or (not cv_eval and label_csv_path is None):
            curr_loggers = list()
        else:
            curr_loggers = loggers
        pl.seed_everything(42, workers=True)
        trainer = pl.Trainer(devices=1,
                             default_root_dir=curr_subfolder_cv,
                             accelerator="auto",
                             max_epochs=config["num_epochs"],
                             callbacks=[MBPlotCallback(curr_subfolder_cv_pred, config)],
                             auto_select_gpus=True,
                             deterministic=True,
                             logger=curr_loggers,
                             limit_val_batches=limit_batches
                             )
        # Load checkpoint if training is continued and ckpt exists
        ckpt_path = os.path.join(curr_subfolder_cv, "last.ckpt")
        trainer.validate(model, dataloader_val, ckpt_path=ckpt_path)
        if not dont_log_splits:
            os.makedirs(curr_subfolder_cv_pred, exist_ok=True)
            np.save(os.path.join(curr_subfolder_cv_pred, "predictions.npy"), model.predictions)
            np.save(os.path.join(curr_subfolder_cv_pred, "targets.npy"), model.targets)
        # To be safe, store a copy
        pred_list.append(model.predictions[None, :, :].copy())
        tar_list.append(model.targets[None, :].copy())
        if dataloader_train_val is not None:
            # Force train val eval setting
            model.is_trainval = True
            trainer.validate(model, dataloader_train_val, ckpt_path=ckpt_path)
            if not dont_log_splits:
                np.save(os.path.join(curr_subfolder_cv_pred, "predictions_train_val.npy"), model.predictions)
                np.save(os.path.join(curr_subfolder_cv_pred, "targets_train_val.npy"), model.targets)
            # To be safe, store a copy
            pred_train_val_list.append(model.predictions[None, :, :].copy())
            tar_train_val_list.append(model.targets[None, :].copy())
            # Reset for next loop
            model.is_trainval = False
        if clearml and not dont_log_splits:
            # Store intermediate preds from CV
            task.upload_artifact(f"pred arr CV{cv_idx+1}", model.predictions)
            task.upload_artifact(f"tar arr CV{cv_idx+1}", model.targets)
    # Aggregate, if at least one split was present
    if model is not None:
        # Aggregate CV predictions: concat for CV mode, average/vote for prediction mode
        pred_train_val_all = None
        tar_train_val_all = None
        pred_list = np.array(pred_list)
        tar_list = np.array(tar_list)
        if cv_eval:
            prefix = "_CV_cat"
            pred_all = np.concatenate(pred_list, axis=0).reshape([-1, pred_list[0].shape[-1]])
            tar_all = np.concatenate(tar_list, axis=0).reshape([-1])
            if eval_on_train:
                pred_train_val_all = np.concatenate(pred_train_val_list, axis=0).reshape([-1, pred_train_val_list[0].shape[-1]])
                tar_train_val_all = np.concatenate(tar_train_val_list, axis=0).reshape([-1])
        else:
            prefix = "_CV_avg"
            pred_all = np.squeeze(np.mean(pred_list, axis=0, keepdims=False), axis=0)
            tar_all = np.squeeze(np.mean(tar_list, axis=0, keepdims=False), axis=0)
        np.save(os.path.join(curr_pred_folder, "predictions_all.npy"), pred_all)
        np.save(os.path.join(curr_pred_folder, "targets_all.npy"), tar_all)
        if eval_on_train:
            np.save(os.path.join(curr_pred_folder, "predictions_train_val_all.npy"), pred_train_val_all)
            np.save(os.path.join(curr_pred_folder, "targets_train_val_all.npy"), tar_train_val_all)
        if clearml:
            # Store intermediate preds from CV
            task.upload_artifact(f"pred arr {prefix}", pred_all)
            task.upload_artifact(f"tar arr {prefix}", tar_all)
        # Perform evaluation through last model
        if cv_eval or label_csv_path is not None:
            model.calc_and_log_metrics(loggers=loggers,
                                       predictions=pred_all,
                                       targets=tar_all,
                                       eval_name="validation",
                                       prefix=prefix,
                                       step=0)
            if pred_train_val_all is not None:
                model.calc_and_log_metrics(loggers=loggers,
                                           predictions=pred_train_val_all,
                                           targets=tar_train_val_all,
                                           eval_name="train_val",
                                           prefix="_CV_cat",
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
