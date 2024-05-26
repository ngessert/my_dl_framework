"""Ensemble script script. Works with or without targets. If there are targets, get metrics as well.

"""

import argparse
import os
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import yaml
from clearml import (Task, TaskTypes)

import my_dl_framework.constants as mc
from my_dl_framework.evaluation.utils import (add_experiments_from_clearml, gather_predictions)
from my_dl_framework.models.pl_class_wrapper import PLClassificationWrapper
from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML


def run_ensemble(config_path: str, clearml: bool, remote: bool):
    """
    Main ensembling execution script
    :param config_path:         Path to ensemble config
    :param clearml:             Whether to use clearml
    :param remote:              Whether this is a remote execution
    :return:
    """
    # Import config
    with open(config_path, encoding='utf-8') as file:
        config = yaml.safe_load(file)
        print(f'Using config {config_path}')

    # for storing ensemble output
    curr_ens_folder = os.path.join(os.path.normpath(config['save_path']), config_path.split(os.sep)[-1] + '_' + datetime.now().strftime('%d-%m-%Y-%H-%M-%S'))
    os.makedirs(curr_ens_folder, exist_ok=True)

    all_eval_folders = config['eval_folders'] or list()

    all_eval_folders = add_experiments_from_clearml(config, all_eval_folders)

    # ClearML
    task = None
    if clearml:
        if remote:
            task = Task.get_task(project_name='RSNABinary', task_name=config_path.split(os.sep)[-1])  # ?
        else:
            task = Task.init(project_name='RSNABinary',
                             task_name=config_path.split(os.sep)[-1],
                             reuse_last_task_id=False,
                             auto_connect_frameworks={'matplotlib': False},
                             task_type=TaskTypes.inference)
        task.connect(config)
        task.add_tags(config['clearml_tags'])

    pred_list, tar_list, training_config = gather_predictions(all_eval_folders)

    if training_config is None:
        raise ValueError('No training config file available - something seems to be wrong with your selected evaluation runs.')

    model = PLClassificationWrapper(config=training_config, training_set_len=0, prefix='dummy_model')
    # Aggregate CV predictions: concat for CV mode, average/vote for prediction mode
    pred_list = np.array(pred_list)
    tar_list = np.array(tar_list)
    if config['comb_type'] == 'mean':
        pred_all = np.mean(pred_list, axis=0, keepdims=False)
        tar_all = np.mean(tar_list, axis=0, keepdims=False)
    else:
        raise ValueError(f"Unsupported ensemble combination type {config['comb_type']}")
    np.save(os.path.join(curr_ens_folder, mc.PRED_FILE_NAME), pred_all)
    np.save(os.path.join(curr_ens_folder, mc.TAR_FILE_NAME), tar_all)
    if clearml:
        # Store intermediate preds from CV
        task.upload_artifact(f"pred arr {config['comb_type']}", os.path.join(curr_ens_folder, mc.PRED_FILE_NAME))
        task.upload_artifact(f"tar arr {config['comb_type']}", os.path.join(curr_ens_folder, mc.TAR_FILE_NAME))
    # Perform evaluation through last model
    if config['ensemble_type'] == 'eval':
        # we also want local logs
        csv_logger = pl.loggers.CSVLogger(curr_ens_folder, name='eval_local_logs')
        if task is not None:
            pl_clearml_logger = PLClearML(task=task, name_base='CV ensemble', title_prefix='_' + config['comb_type'])
        else:
            pl_clearml_logger = None
        loggers = [csv_logger]
        if pl_clearml_logger is not None:
            loggers.append(pl_clearml_logger)
        pl.seed_everything(42, workers=True)
        model.calc_and_log_metrics(loggers=loggers, predictions=pred_all, targets=tar_all, eval_name='ensemble', prefix='_' + config['comb_type'], step=0)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-f', '--config_path', type=str, help='Path to ensemble config', required=True)
    argparser.add_argument('-cl', '--clearml', action='store_true', help='Whether to use clearml for logging results', required=False)
    argparser.add_argument('-r', '--remote', action='store_true', help='Whether remote execution is done', required=False)

    args = argparser.parse_args()
    print(f'Args: {args}')
    run_ensemble(config_path=args.config_path, clearml=args.clearml, remote=args.remote)
