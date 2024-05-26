"""Main Training script with pytorch lightning.
Example:
    python my_dl_framework\\training\\train_pl.py --config=C:\\sources\\my_dl_framework\\configs\\test_config.yaml -cl clearml
"""

import argparse
import json
import os
from typing import Union

import lightning.pytorch as pl
import yaml
from lightning.pytorch.profilers import SimpleProfiler

import my_dl_framework.constants as mc
from my_dl_framework.models.pl_class_wrapper import PLClassificationWrapper
from my_dl_framework.training.utils import (get_callbacks, get_ckpt_path, get_dataloaders, setup_clearml_training, setup_data, setup_local_folders)
from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML


def run_training(config_path: str, clearml: bool, num_gpus: Union[int, str, None], multi_gpu_strat: Union[str, None], fast_dev_run: int):
    """
    Run pytorch lightning training
    :param config_path:              Path to config file
    :param clearml:             Whther to use clearml
    :param num_gpus:            Number of GPUs to use
    :param multi_gpu_strat:     Pytorch lightning GPU strat (e.g. dpp)
    :param fast_dev_run:        Pytorch lighning dev-run option
    :return:
    """
    # Config is used everywhere
    with open(config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
        print(f'Using config {config_path}')

    # ClearML requires setup
    task = None
    task_prev = None
    if clearml:
        task, task_prev = setup_clearml_training(config, config_path)

    images_path, labels_path, cv_file_path = setup_data(config)

    # Data splits are predefined in a file
    with open(os.path.normpath(cv_file_path), encoding='utf-8') as file:
        data_split = json.load(file)
    training_subsets = data_split['training_splits']
    validation_subsets = data_split['validation_splits']

    curr_subfolder = setup_local_folders(config, cv_file_path, config_path, clearml, task)

    # Main CV loop
    for cv_idx, (subset_train, subset_val) in enumerate(zip(training_subsets, validation_subsets)):
        prefix = None
        if config['run_cv_subset'] is not None:
            # Skip CV folds that are not selected
            if cv_idx not in config['run_cv_subset']:
                continue
            if len(config['run_cv_subset']) == 1:
                prefix = ''
        prefix = mc.CV_PREFIX + str(cv_idx + 1) if prefix is None else prefix
        print(f'Starting fold number {cv_idx+1} / {len(training_subsets)}')
        curr_subfolder_cv = os.path.join(curr_subfolder, 'CV_' + str(cv_idx + 1))
        os.makedirs(curr_subfolder_cv, exist_ok=True)
        # Data
        dataloader_train, dataloader_val, dataloader_train_val = get_dataloaders(config, images_path, labels_path, subset_train, subset_val)
        model = PLClassificationWrapper(config=config, training_set_len=len(dataloader_train.dataset), prefix=prefix)
        callback_list = get_callbacks(config, prefix, curr_subfolder_cv, iter_per_epoch=20)  # len(dataloader_train))
        # we also want local logs
        csv_logger = pl.loggers.CSVLogger(curr_subfolder_cv, name='local_logs')
        if task is not None:
            pl_clearml_logger = PLClearML(task=task, name_base=prefix, title_prefix=mc.CV_PREFIX)
        else:
            pl_clearml_logger = None
        pl.seed_everything(42, workers=True)
        trainer = pl.Trainer(
            devices=num_gpus,
            check_val_every_n_epoch=config['validate_every_x_epochs'],
            default_root_dir=curr_subfolder_cv,
            accelerator='auto',
            max_epochs=config['num_epochs'],
            strategy=multi_gpu_strat,
            precision=32,  # "16-mixed" if torch.cuda.is_available() else "32",
            deterministic=True,
            fast_dev_run=fast_dev_run,
            callbacks=callback_list,
            logger=[pl_clearml_logger, csv_logger] if pl_clearml_logger is not None else csv_logger,
            limit_train_batches=50,
            limit_val_batches=config['max_num_batches_val'],
            log_every_n_steps=config['loss_log_freq'],
            profiler=None,  # SimpleProfiler(dirpath="./", filename="simple_profiler"),
        )
        # Load checkpoint if training is continued and ckpt exists
        ckpt_path = get_ckpt_path(config, curr_subfolder_cv, cv_idx, task_prev)
        trainer.fit(model, dataloader_train, [dataloader_val] if dataloader_train_val is None else [dataloader_val, dataloader_train_val], ckpt_path=ckpt_path)
    if task is not None:
        task.close()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', type=str, help='Config path', required=True)
    argparser.add_argument('-cl', '--clearml', action='store_true', help='Whether to use clearml', required=False)
    argparser.add_argument('-ng', '--num_gpus', default='auto', help='Number of GPUs to use')
    argparser.add_argument('-st', '--multi_gpu_strat', type=str, default='auto', help='MultiGPU strat, e.g. ddp')
    argparser.add_argument('-fd', '--fast_dev_run', type=int, default=None, help='test only x batches')
    args = argparser.parse_args()
    print(f'Args: {args}')
    run_training(config_path=args.config, clearml=args.clearml, num_gpus=args.num_gpus, multi_gpu_strat=args.multi_gpu_strat, fast_dev_run=args.fast_dev_run)
