"""Prediction script. Works with or without targets. If there are targets, get metrics as well.

"""

import argparse
import shutil
from datetime import datetime
import yaml
import os
import numpy as np
from glob import glob
from torch.utils.data import DataLoader
from typing import Optional

from my_dl_framework.data.get_dataset import get_dataset
from my_dl_framework.evaluation.utils import define_tta_options, get_data_split, get_clearml_tags, setup_clearml, \
    setup_folders_and_clearml, run_checks, setup_loggers, aggregate_eval_results
from my_dl_framework.models.pl_class_wrapper import PLClassificationWrapper
import pytorch_lightning as pl

from my_dl_framework.utils.pytorch_lightning.minibatch_plot_callback import MBPlotCallback
import my_dl_framework.constants as mc


def run_prediction(folder: str,
                   clearml_id: Optional[str],
                   clearml: bool,
                   eval_on_train: bool,
                   cv_eval: bool,
                   image_path: str,
                   limit_batches: int,
                   dont_log_splits: bool,
                   label_csv_path: str,
                   tta_flip_horz: bool,
                   tta_flip_vert: bool,
                   tta_multi_eq_crop: int):

    tta_options, batch_size_factor, collate_fn = define_tta_options(tta_flip_horz, tta_flip_vert, tta_multi_eq_crop)

    image_path = os.path.normpath(image_path)
    setup_folders_and_clearml(folder, clearml_id)
    # Import config
    with open(os.path.join(folder, mc.CFG_NAME), encoding="utf-8") as file:
        config = yaml.safe_load(file)
        print(f'Using config {os.path.join(folder, mc.CFG_NAME)}')
    data_split = get_data_split(cv_eval, folder)
    # ClearML
    tags = get_clearml_tags(cv_eval, eval_on_train, limit_batches, dont_log_splits, tta_flip_horz, tta_flip_vert,
                            tta_multi_eq_crop)
    arg_dict = {"folder": folder, "clearml_id": clearml_id, "eval_on_train": eval_on_train,
                "cv_eval": cv_eval, "image_path": image_path, "limit_batches": limit_batches,
                "dont_log_splits": dont_log_splits}
    task = setup_clearml(clearml, tags, arg_dict, folder)
    # for storing predictions, there may be many evaluations
    curr_pred_folder = os.path.join(folder, "pred_and_eval_" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S"))
    os.makedirs(curr_pred_folder, exist_ok=True)
    # copy config for ensembling later on
    shutil.copy(os.path.join(folder, mc.CFG_NAME), os.path.join(curr_pred_folder, "training_config.yaml"))
    pred_list = list()
    pred_train_val_list = list()
    tar_list = list()
    tar_train_val_list = list()
    model = None
    loggers = list()
    for curr_subfolder_cv in glob(os.path.join(folder, "CV_*")):
        if os.path.isfile(curr_subfolder_cv):
            continue
        cv_idx = int(curr_subfolder_cv.split("CV_")[-1]) - 1
        continue_loop, path_to_label_csv, subset_val, subset_train = run_checks(curr_subfolder_cv, data_split,
                                                                                label_csv_path, cv_eval, image_path,
                                                                                config, cv_idx)
        if continue_loop:
            continue
        dataset_val = get_dataset(config=config,
                                  image_dir=image_path,
                                  path_to_label_csv=path_to_label_csv,
                                  subset=subset_val,
                                  tta_options=tta_options,
                                  is_training=False)
        print(f'Size validation dataset {len(dataset_val)}')
        batch_size_adj = config["batch_size"] // batch_size_factor
        if batch_size_adj <= 0:
            batch_size_adj = 1
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=batch_size_adj, shuffle=False,
                                    num_workers=8, pin_memory=True, collate_fn=collate_fn)
        dataloader_train_val = None
        if eval_on_train and cv_eval:
            dataset_train_val = get_dataset(config=config,
                                            image_dir=image_path,
                                            path_to_label_csv=path_to_label_csv,
                                            subset=subset_train,
                                            tta_options=tta_options,
                                            is_training=False)
            print(f'Size validation dataset {len(dataset_train_val)}')
            dataloader_train_val = DataLoader(dataset=dataset_train_val, batch_size=batch_size_adj, shuffle=False,
                                              num_workers=8, pin_memory=True, collate_fn=collate_fn)
        model = PLClassificationWrapper(config=config,
                                        training_set_len=len(dataset_val),
                                        prefix=mc.CV_PREFIX + str(cv_idx+1))
        # grab from model and save
        curr_subfolder_cv_pred = os.path.join(curr_pred_folder, "CV_" + str(cv_idx + 1))
        # we also want local logs
        loggers, curr_loggers = setup_loggers(task, curr_subfolder_cv_pred, cv_idx, dont_log_splits, cv_eval, label_csv_path)
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
        ckpt_path = os.path.join(curr_subfolder_cv, mc.CKPT_NAME)
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
    # Aggregate, if at least one split was present
    aggregate_eval_results(cv_eval, eval_on_train, pred_list, tar_list, pred_train_val_list, tar_train_val_list,
                           curr_pred_folder, model, label_csv_path, loggers, clearml, task)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--folder', type=str, help='Folder path', required=False)
    argparser.add_argument('-cid', '--clearml_id', type=str, help='ClearML ID of experiment', required=False)
    argparser.add_argument('-cl', '--clearml', action="store_true", help='Whether to use clearml for logging results', required=False)
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
                   eval_on_train=args.eval_on_train,
                   cv_eval=args.cv_eval,
                   image_path=args.image_path,
                   limit_batches=args.limit_batches,
                   dont_log_splits=args.dont_log_splits,
                   label_csv_path=args.label_csv_path,
                   tta_flip_horz=args.tta_flip_horz,
                   tta_flip_vert=args.tta_flip_vert,
                   tta_multi_eq_crop=args.tta_multi_eq_crop)
