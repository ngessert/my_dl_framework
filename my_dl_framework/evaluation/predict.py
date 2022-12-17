"""Prediction script. Works with or without targets. If there are targets, get metrics as well.
Options:
    --folder=<folder>     folder to run prediction on
    -cl --clearml         use clearml.
Example:
    python my_dl_framework\\evaluation\\predict.py --folder=C:\\data\\RSNA_challenge\\experiments\\test_config.yaml20-02-2022-20-38-12 -cl clearml
"""

import argparse
import json
import yaml
import os
import numpy as np
import torch
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader
from my_dl_framework.training.utils import get_dataset, get_model, get_and_log_metrics_classification
from clearml import Task, TaskTypes


def run_prediction(args):
    if not os.path.exists(args.folder):
        raise ValueError("Provided subfolder does not exist.")
    all_files = glob(os.path.join(args.folder, "*"))
    config_file = None
    for file in all_files:
        if file.endswith(".yml") or file.endswith(".yaml"):
            config_file = file
            break
    if config_file is None:
        raise ValueError("Could not find a config in the folder")
    # Import config
    with open(config_file) as f:
        config = yaml.safe_load(f)
        print(f'Using config {config_file}')
    # ClearML
    if args.clearml:
        if args.remote:
            task = Task.get_task(project_name="RSNABinary", task_name=config_file.split(os.sep)[-1])  # ?
        else:
            task = Task.init(project_name='RSNABinary',
                             task_name=config_file.split(os.sep)[-1],
                             reuse_last_task_id=False,
                             auto_connect_frameworks={'matplotlib': False},
                             task_type=TaskTypes.inference
                             )
        task.connect(config)
        logger = task.get_logger()
    else:
        task = None
        logger = None
    # Setup CV
    with open(config["data_split_file"]) as f:  #os.path.join(config['base_path']
        data_split = json.load(f)
    training_subsets = data_split["training_splits"]
    validation_subsets = data_split["validation_splits"]
    for idx, (subset_train, subset_val) in enumerate(zip(training_subsets, validation_subsets)):
        print(f'Starting fold number {idx+1}/{len(training_subsets)}')
        curr_subfolder_cv = os.path.join(args.folder, "CV_" + str(idx+1))
        # Load model
        model = get_model(config=config)
        model = model.cuda()
        # Get model path
        all_files = glob(curr_subfolder_cv + os.sep + "*")
        model_file_name = None
        for file_name in all_files:
            if "model" in file_name and "latest" in file_name:
                model_file_name = file_name
        if model_file_name is None:
            raise ValueError(f"Model file not found in folder {curr_subfolder_cv}")
        state_model = torch.load(model_file_name)
        model.load_state_dict(state_model['state_dict'])
        dataset_val = get_dataset(config=config, image_dir=config["training_image_dir"], subset=subset_val,
                                  is_training=False)
        print(f'Size validation dataset {len(dataset_val)}')
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=config["batch_size"], shuffle=True,
                                    num_workers=8, pin_memory=True)
        _, pred_val, tar_val = get_and_log_metrics_classification(eval_name="validation_predict", dataloader=dataloader_val, model=model,
                                                                  config=config, logger=logger, epoch=0,
                                                                  curr_subfolder=curr_subfolder_cv,
                                                                  use_clearml=args.clearml)
        if args.eval_on_train:
            dataset_train_val = get_dataset(config=config, image_dir=config["training_image_dir"], subset=subset_train,
                                            is_training=False)
            print(f'Size train dataset {len(dataset_train_val)}')
            dataloader_train_val = DataLoader(dataset=dataset_train_val, batch_size=config["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)
            _, pred_val, tar_val = get_and_log_metrics_classification(eval_name="validation_on_train_predict", dataloader=dataloader_train_val,
                                                                      model=model,
                                                                      config=config, logger=logger, epoch=0,
                                                                      curr_subfolder=curr_subfolder_cv,
                                                                      use_clearml=args.clearml)
        if args.test_path is not None:
            dataset_test = get_dataset(config=config, image_dir=args.test_path, subset=None,
                                       is_training=False)
            print(f'Size test dataset {len(dataset_test)}')
            dataloader_test = DataLoader(dataset=dataset_test, batch_size=config["batch_size"], shuffle=False,
                                         num_workers=8, pin_memory=True)
            _, pred_val, tar_val = get_and_log_metrics_classification(eval_name="validation_on_test_predict", dataloader=dataloader_test,
                                                                      model=model,
                                                                      config=config, logger=logger, epoch=0,
                                                                      curr_subfolder=curr_subfolder_cv,
                                                                      use_clearml=args.clearml)
            final_df = pd.DataFrame()
            final_df["patientID"] = pred_dict["patientID"]



if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--folder', type=str, help='Folder path', required=True)
    argparser.add_argument('-cl', '--clearml', type=bool, default=False, help='Whether to use clearml', required=False)
    argparser.add_argument('-r', '--remote', type=bool, default=False, help='Whether remote execution is done', required=False)
    argparser.add_argument('-etr', '--eval_on_train', type=bool, default=False, help='Whether to evaluate on the training set', required=False)
    argparser.add_argument('-t', '--test_path', type=str, default=None, help='Test image path for prediction', required=False)
    args = argparser.parse_args()
    print(f'Args: {args}')
    run_prediction(args)
