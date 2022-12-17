"""Training script.
Options:
    -h --help            show this message and exit.
    --config=<config>    config name to run.
    -cl --clearml         use clearml.
Example:
    python my_dl_framework\\training\\train.py --config=C:\\sources\\my_dl_framework\\configs\\test_config.yaml -cl clearml
"""

import argparse
import json
import yaml
import os
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from my_dl_framework.training.utils import get_dataset, get_model, get_lossfunction, get_optimizer, get_lr_scheduler, save_optimizer_and_model, get_and_log_metrics_classification, NumpyEncoder, plot_example_batch
from clearml import Task
from datetime import datetime


def run_training(args):
    # Import config
    with open(args.config) as f:
        config = yaml.safe_load(f)
        print(f'Using config {args.config}')
    # ClearML
    if args.clearml:
        if args.remote:
            task = Task.get_task(project_name="RSNABinary", task_name=args.config.split(os.sep)[-1])  # ?
        else:
            task = Task.init(project_name='RSNABinary',
                             task_name=args.config.split(os.sep)[-1],
                             reuse_last_task_id=False,
                             auto_connect_frameworks={'matplotlib': False}
                             )
        task.connect(config)
        logger = task.get_logger()
    else:
        task = None
        logger = None
    # Run Training
    # Setup CV
    with open(config["data_split_file"]) as f:
        data_split = json.load(f)
    training_subsets = data_split["training_splits"]
    validation_subsets = data_split["validation_splits"]
    curr_subfolder = os.path.join(config['base_path'], "experiments", args.config.replace("\\", "/").split("/")[-1])
    if os.path.exists(curr_subfolder) and not config["continue_training"]:
        curr_subfolder += datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    for cv_idx, (subset_train, subset_val) in enumerate(zip(training_subsets, validation_subsets)):
        print(f'Starting fold number {cv_idx+1}/{len(training_subsets)}')
        curr_subfolder_cv = os.path.join(curr_subfolder, "CV_" + str(cv_idx+1))
        # Data
        dataset_train = get_dataset(config=config, image_dir=config["training_image_dir"], subset=subset_train,
                                    is_training=True)
        print(f'Size training dataset {len(dataset_train)}')

        dataloader_train = DataLoader(dataset=dataset_train, batch_size=config["batch_size"], shuffle=True,
                                      num_workers=8, pin_memory=True,
                                      worker_init_fn=None)
        dataset_val = get_dataset(config=config, image_dir=config["training_image_dir"], subset=subset_val,
                                  is_training=False)
        print(f'Size validation dataset {len(dataset_val)}')

        dataloader_val = DataLoader(dataset=dataset_val, batch_size=config["batch_size"], shuffle=True,
                                    num_workers=8, pin_memory=True)
        if config['validate_on_train_set']:
            dataset_train_val = get_dataset(config=config, image_dir=config["training_image_dir"], subset=subset_train,
                                            is_training=False)
            print(f'Size validation dataset {len(dataset_train_val)}')
            dataloader_train_val = DataLoader(dataset=dataset_train_val, batch_size=config["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)
        else:
            dataloader_train_val = None
        # Model
        model = get_model(config=config)
        model = model.cuda() if torch.cuda.is_available() else model
        # Loss function/optimizer
        loss_function = get_lossfunction(config=config)
        optimizer = get_optimizer(config=config, model=model)
        lr_scheduler = get_lr_scheduler(config=config, optimizer=optimizer) if config['lr_scheduler'] is not None\
            else None
        # Folder setup/perhaps load previous state
        if os.path.exists(curr_subfolder_cv) and config["continue_training"]:
            # Check if checkpoint exists
            all_files = glob(os.path.join(curr_subfolder_cv, "*"))
            max_epoch = -1
            optimizer_path = None
            for file_name in all_files:
                if "optimizer_ckpt" in file_name:
                    curr_epoch = int(os.path.normpath(file_name).split(os.path.sep)[-1].split(".")[0].split("_")[-1])
                    if curr_epoch > max_epoch:
                        optimizer_path = file_name
                        max_epoch = curr_epoch
            # Load optimizer and model
            if optimizer_path is not None:
                state_opt = torch.load(optimizer_path)
                optimizer.load_state_dict(state_opt['state_dict'])
                state_model = torch.load(optimizer_path.replace("optimizer", "model"))
                model.load_state_dict(state_model['state_dict'])
                start_epoch = max_epoch
                # Load metric tracking
                if os.path.exists(os.path.join(curr_subfolder_cv, "training_metrics.json")):
                    with open(os.path.join(curr_subfolder_cv, "training_metrics.json")) as f:
                        metrics_train_all = json.load(f)
                        for key in metrics_train_all:
                            metrics_train_all[key] = np.asarray(metrics_train_all[key])
                else:
                    metrics_train_all = dict()
            else:
                start_epoch = 0
                metrics_train_all = dict()
        else:
            metrics_train_all = dict()
            start_epoch = 0
            os.makedirs(curr_subfolder_cv, exist_ok=False)
        # Training
        for epoch in tqdm(range(start_epoch, config['num_epochs']), disable=args.clearml):
            np.random.seed(np.random.get_state()[1][0] + epoch)
            model.train()
            for batch_idx, (indices, images, targets) in tqdm(enumerate(dataloader_train), disable=args.clearml):
                if epoch == 0 and batch_idx <= config["num_batch_examples"]:
                    plot_example_batch(images, targets, batch_idx, curr_subfolder_cv, config)
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = targets.cuda()
                optimizer.zero_grad()
                # print("train tar", targets)
                with torch.set_grad_enabled(True):
                    outputs = model(images)
                    loss = loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()
                # Track loss
                if batch_idx % config["loss_log_freq"] == 0:
                    if "train_loss" not in metrics_train_all:
                        metrics_train_all["train_loss"] = np.array([[epoch * len(dataloader_train) + batch_idx,
                                                               loss.detach().cpu().numpy()]])
                    else:
                        metrics_train_all["train_loss"] = np.concatenate((metrics_train_all["train_loss"], np.array(
                            [[epoch*len(dataloader_train) + batch_idx, loss.detach().cpu().numpy()]])), axis=0)
                    if args.clearml:
                        logger.report_scalar(title="Loss", series="Train Loss", value=loss.detach().cpu().numpy(),
                                             iteration=epoch * len(dataloader_train) + batch_idx)
            if lr_scheduler is not None:
                lr_scheduler.step()
            print(f'Fold {cv_idx+1}/{len(training_subsets)} Epoch {epoch}/{config["num_epochs"]} completed. '
                  f'Last train loss: {loss.detach().cpu().numpy()}')
            # Save
            save_optimizer_and_model(optimizer=optimizer, model=model, curr_subfolder=curr_subfolder_cv,
                                     epoch=epoch, prefix="last_")
            # Save
            with open(os.path.join(curr_subfolder_cv, "training_metrics.json"), "w") as f:
                json.dump(metrics_train_all, f, cls=NumpyEncoder)
            # Validate in between
            if epoch % config['validate_every_x_epochs'] == 0:
                metrics_all, _, _ = get_and_log_metrics_classification(eval_name="validation", dataloader=dataloader_val, model=model,
                                                                       config=config, logger=logger, epoch=epoch, curr_subfolder=curr_subfolder_cv,
                                                                       use_clearml=args.clearml)
                # Determine if there is a new best validation model
                if metrics_all[metrics_all["best_epoch"]][config['val_best_metric']][-1] < metrics_all[epoch][config['val_best_metric']][-1]:
                    print(
                        f'New best epoch {epoch} with {metrics_all[epoch][config["val_best_metric"]][-1]} (before: {metrics_all[metrics_all["best_epoch"]][config["val_best_metric"]][-1]})')
                    metrics_all["best_epoch"] = epoch
                    save_optimizer_and_model(optimizer=optimizer, model=model, curr_subfolder=curr_subfolder_cv,
                                             epoch=epoch, prefix="best_")
                if config['validate_on_train_set']:
                    _ = get_and_log_metrics_classification(eval_name="training", dataloader=dataloader_train_val,
                                                           model=model,
                                                           config=config, logger=logger, epoch=epoch,
                                                           curr_subfolder=curr_subfolder_cv,
                                                           use_clearml=args.clearml)
        # Log artifacts in clearml
        if args.clearml:
            if os.path.isfile(os.path.join(curr_subfolder_cv, "training_metrics.json")):
                task.upload_artifact(name=f"training_metrics_cv{cv_idx}.json",
                                     artifact_object=os.path.join(curr_subfolder_cv, "training_metrics.json"))
            if os.path.isfile(os.path.join(curr_subfolder_cv, "validation_metrics.json")):
                task.upload_artifact(name=f"validation_metrics_cv{cv_idx}.json",
                                     artifact_object=os.path.join(curr_subfolder_cv, "validation_metrics.json"))
            # TODO: upload model and optimizer
    print("Training Completed")
    if args.clearml:
        task.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', type=str, help='Config path', required=True)
    argparser.add_argument('-cl', '--clearml', type=bool, default=False, help='Whether to use clearml', required=False)
    argparser.add_argument('-r', '--remote', type=bool, default=False, help='Whether remote execution is done', required=False)
    args = argparser.parse_args()
    print(f'Args: {args}')
    run_training(args)
