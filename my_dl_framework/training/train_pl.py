"""Main Training script with pytorch lightning.
Example:
    python my_dl_framework\\training\\train_pl.py --config=C:\\sources\\my_dl_framework\\configs\\test_config.yaml -cl clearml
"""

import argparse
import json
import os
import shutil
from datetime import datetime
import yaml
from torch.utils.data import DataLoader

from clearml import Task
import pytorch_lightning as pl
from typing import Union

from my_dl_framework.data.get_dataset import get_dataset
from my_dl_framework.models.pl_wrapper import PLClassificationWrapper
from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML
from my_dl_framework.utils.pytorch_lightning.minibatch_plot_callback import MBPlotCallback


def run_training(config_path: str,
                 clearml: bool,
                 num_gpus: Union[int, None],
                 multi_gpu_strat: Union[str, None],
                 remote: bool,
                 fast_dev_run: bool):
    """
    Run pytorch lightning training
    :param config_path:              Path to config file
    :param clearml:             Whther to use clearml
    :param num_gpus:            Number of GPUs to use
    :param multi_gpu_strat:     Pytorch lightning GPU strat (e.g. dpp)
    :param remote:              Whether job is being executed remotely (auto-set by clearml)
    :param fast_dev_run:        Pytorch lighning dev-run option
    :return:
    """
    # Import config
    with open(config_path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
        print(f'Using config {config_path}')
    # ClearML
    task = None
    task_prev = None
    if clearml:
        if remote:
            task = Task.get_task(project_name="RSNABinary", task_name=config_path.split(os.sep)[-1])  # ?
        else:
            task = Task.init(project_name='RSNABinary',
                             task_name=config_path.split(os.sep)[-1],
                             reuse_last_task_id=False,
                             auto_connect_frameworks={'matplotlib': False}
                             )
            if config["continue_training_from_clearml"] is not None:
                task_prev = Task.get_task(task_id=config["continue_training_from_clearml"])
                print(f'Use ckpts from task {config["continue_training_from_clearml"]}')
        task.connect(config)
    # Run Training
    # Setup CV
    with open(os.path.normpath(config["data_split_file"]), encoding="utf-8") as file:
        data_split = json.load(file)
    training_subsets = data_split["training_splits"]
    validation_subsets = data_split["validation_splits"]
    curr_subfolder = os.path.join(os.path.normpath(config['base_path']), "experiments", config_path.replace("\\", "/").split("/")[-1])
    if config["continue_training"] is not None:
        curr_subfolder += config["continue_training"]
    else:
        curr_subfolder += datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    for cv_idx, (subset_train, subset_val) in enumerate(zip(training_subsets, validation_subsets)):
        if config["run_cv_subset"] is not None:
            # Skip CV folds that are not selected
            if cv_idx not in config["run_cv_subset"]:
                continue
        print(f'Starting fold number {cv_idx+1}/{len(training_subsets)}')
        curr_subfolder_cv = os.path.join(curr_subfolder, "CV_" + str(cv_idx+1))
        os.makedirs(curr_subfolder_cv, exist_ok=True)
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

        dataloader_val = DataLoader(dataset=dataset_val, batch_size=config["batch_size"], shuffle=False,
                                    num_workers=8, pin_memory=True)
        if config['validate_on_train_set']:
            dataset_train_val = get_dataset(config=config, image_dir=config["training_image_dir"], subset=subset_train,
                                            is_training=False)
            print(f'Size validation dataset {len(dataset_train_val)}')
            dataloader_train_val = DataLoader(dataset=dataset_train_val, batch_size=config["batch_size"], shuffle=False,
                                              num_workers=8, pin_memory=True)
        else:
            dataloader_train_val = None
        model = PLClassificationWrapper(config=config,
                                        training_set_len=len(dataset_train))
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=config["val_best_metric"],
            save_last=True,
            save_top_k=1,
            dirpath=curr_subfolder_cv,
            filename='train-{epoch:02d}-{' + config["val_best_metric"] + ':.4f}',
            mode=config["val_best_target"],
            auto_insert_metric_name=True,
            every_n_epochs=config["ckpt_every_n_epochs"],
        )
        # we also want local logs
        csv_logger = pl.loggers.CSVLogger(curr_subfolder_cv, name="local_logs")
        if task is not None:
            pl_clearml_logger = PLClearML(task=task, name_base="CV" + str(cv_idx+1))
        else:
            pl_clearml_logger = None
        pl.seed_everything(42, workers=True)
        trainer = pl.Trainer(devices=num_gpus,
                             check_val_every_n_epoch=config["validate_every_x_epochs"],
                             default_root_dir=curr_subfolder_cv,
                             accelerator="auto",
                             max_epochs=config["num_epochs"],
                             auto_select_gpus=True if num_gpus is None else False,
                             strategy=multi_gpu_strat,
                             precision=16 if num_gpus is not None else None,
                             deterministic=True,
                             fast_dev_run=fast_dev_run,
                             callbacks=[checkpoint_callback, MBPlotCallback(curr_subfolder_cv, config)],
                             logger=[pl_clearml_logger, csv_logger] if pl_clearml_logger is not None else csv_logger,
                             limit_train_batches=20,
                             limit_val_batches=config["max_num_batches_val"],
                             log_every_n_steps=config["loss_log_freq"]
                             )
        # Load checkpoint if training is continued and ckpt exists
        if config["continue_training"] is not None and os.path.isfile(os.path.join(curr_subfolder_cv, "last.ckpt")):
            ckpt_path = os.path.join(curr_subfolder_cv, "last.ckpt")
        elif task_prev is not None:
            print("No local ckpt found, getting ckpt from ClearML")
            ckpt_path = os.path.join(curr_subfolder_cv, "last.ckpt")
            ckpt_path_tmp = task_prev.artifacts[f"model-CV{cv_idx+1}-latest"].get_local_copy()
            shutil.copy(ckpt_path_tmp, ckpt_path)
        else:
            ckpt_path = None
        trainer.fit(model, dataloader_train, 
                    [dataloader_val] if dataloader_train_val is None else [dataloader_val, dataloader_train_val],
                    ckpt_path=ckpt_path)
    if task is not None:
        task.close()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', type=str, help='Config path', required=True)
    argparser.add_argument('-cl', '--clearml', type=bool, default=False, help='Whether to use clearml', required=False)
    argparser.add_argument('-ng', '--num_gpus', type=int, default=1, help="Number of GPUs to use")
    argparser.add_argument('-st', '--multi_gpu_strat', type=str, default=None, help="MultiGPU strat, e.g. ddp")
    argparser.add_argument('-r', '--remote', type=bool, default=False, help='Whether remote execution is done remote', required=False)
    argparser.add_argument('-fd', '--fast_dev_run', type=int, default=None, help="test only x batches")
    args = argparser.parse_args()
    print(f'Args: {args}')
    run_training(config_path=args.config,
                 clearml=args.clearml,
                 num_gpus=args.num_gpus,
                 multi_gpu_strat=args.multi_gpu_strat,
                 remote=args.remote,
                 fast_dev_run=args.fast_dev_run)
