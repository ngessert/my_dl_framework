"""
Training utility stuff
"""
import json
import os
import shutil
from datetime import datetime
from typing import (
    Dict,
    List,
    Tuple
)

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import plotly
import torch
from clearml import (
    Dataset,
    Logger,
    Task
)
from clearml.config import running_remotely
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    RichProgressBar
)
from torch.utils.data import DataLoader

import my_dl_framework.constants as mc
from my_dl_framework.data.get_dataset import get_pt_dataset
from my_dl_framework.evaluation.utils import validate_model_classification
from my_dl_framework.utils.pytorch_lightning.minibatch_plot_callback import (
    MBPlotCallback
)


def get_and_log_metrics_classification(eval_name: str, dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, config: Dict, logger: Logger,
                                       epoch: int, curr_subfolder: str, use_clearml: bool) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Function that calls the metric evaluation and then logs it ClearML and disk
    :param eval_name:       Name of the evaluation, e.g. train or test
    :param dataloader:      Dataloader to draw samples from
    :param model:           Model
    :param config:          Config dict
    :param logger:          ClearML logger
    :param epoch:           Epoch for logging
    :param curr_subfolder:  Subfolder where to store local metrics
    :param use_clearml:     Whether clearml is used
    :return:                Metrics, predictions, and targets
    """
    print(f'Validating on set {eval_name} of length {len(dataloader)}')
    metrics, plots, pred, tar = validate_model_classification(model=model,
                                                              dataloader=dataloader,
                                                              config=config,
                                                              max_num_batches=config['max_num_batches_val'],
                                                              use_cleaml=use_clearml)
    # convert to pandas DF
    metric_df = pd.DataFrame.from_dict(metrics)
    metric_df['classes'] = config['data']['class_names'] + ['Avg']
    metric_df.set_index('classes', inplace=True)
    print('Metrics' + '-' * 20)
    print(metric_df)
    print('-' * 25)
    if use_clearml:
        logger.report_table(title='Metrics', series='Metrics', iteration=epoch, table_plot=metric_df)
        for metric_name, metric_val in metrics.items():
            logger.report_scalar(title=metric_name, series=eval_name + ' ' + metric_name, value=np.mean(metric_val), iteration=epoch)
        for plot_name, plot in plots.items():
            # Plotly figure
            if isinstance(plot, plotly.graph_objs.Figure):
                logger.report_plotly(title=eval_name + ' ' + plot_name, series=eval_name + ' ' + plot_name, figure=plot, iteration=epoch)
    # Save metrics
    if os.path.exists(os.path.join(curr_subfolder, eval_name + mc.METRICS_FILE)):
        with open(os.path.join(curr_subfolder, eval_name + mc.METRICS_FILE), encoding='utf-8') as file:
            metrics_all = json.load(file)
        for key in metrics_all:
            metrics_all[key] = np.asarray(metrics_all[key])
        metrics_all[epoch] = metrics
    else:
        metrics_all = dict()
        metrics_all['best_epoch'] = epoch
        metrics_all[epoch] = metrics
    # Save
    with open(os.path.join(curr_subfolder, eval_name + mc.METRICS_FILE), encoding='utf-8') as file:
        json.dump(metrics_all, file, cls=NumpyEncoder)
    return metrics_all, pred, tar


class NumpyEncoder(json.JSONEncoder):
    """ Numpy encoder for JSON saving.
    """

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)


def get_dataloaders(config: Dict, images_path: str, labels_path: str, subset_train: List[str],
                    subset_val: List[str]) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Sets up the dataloaders
    :param config:          Full config
    :param images_path:     Path to images folder
    :param labels_path:     Path to labels CSV
    :param subset_train:    List of training cases
    :param subset_val:      List of validation cases
    :return:
    """
    dataset_train = get_pt_dataset(config=config,
                                   image_dir=os.path.normpath(images_path),
                                   path_to_label_csv=os.path.normpath(labels_path),
                                   subset=subset_train,
                                   is_training=True,
                                   tta_options=None)
    print(f'Size training dataset {len(dataset_train)}')

    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=config['batch_size'],
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True,
                                  persistent_workers=True,
                                  worker_init_fn=None)
    dataset_val = get_pt_dataset(config=config,
                                 image_dir=os.path.normpath(images_path),
                                 path_to_label_csv=os.path.normpath(labels_path),
                                 subset=subset_val,
                                 is_training=False,
                                 tta_options=None)
    print(f'Size validation dataset {len(dataset_val)}')

    dataloader_val = DataLoader(dataset=dataset_val, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    if config['validate_on_train_set']:
        dataset_train_val = get_pt_dataset(config=config,
                                           image_dir=os.path.normpath(images_path),
                                           path_to_label_csv=os.path.normpath(labels_path),
                                           subset=subset_train,
                                           is_training=False,
                                           tta_options=None)
        print(f'Size validation dataset {len(dataset_train_val)}')
        dataloader_train_val = DataLoader(dataset=dataset_train_val,
                                          batch_size=config['batch_size'],
                                          shuffle=False,
                                          num_workers=4,
                                          pin_memory=True,
                                          persistent_workers=True)
    else:
        dataloader_train_val = None
    return dataloader_train, dataloader_val, dataloader_train_val


def setup_data(config: Dict) -> Tuple[str, str, str]:
    """
    Grabs the data from a ClearML dataset.

    :param config:      Full config
    :return:
    """
    cl_dataset = Dataset.get(config['cl_dataset_id'], alias='used_dataset')
    dataset_metadata = cl_dataset.get_metadata().to_dict()['0']
    base_path = cl_dataset.get_local_copy(use_soft_links=False)
    images_path = str(os.path.join(base_path, dataset_metadata['images_dir']))
    if 'labels_dir' not in dataset_metadata:
        raise ValueError(f"Dataset {cl_dataset.name} (ID: {cl_dataset.id} does not have labels.")
    labels_path = str(os.path.join(base_path, dataset_metadata['labels_dir']))
    if 'csv_name' in dataset_metadata:
        labels_path = str(os.path.join(labels_path, dataset_metadata['csv_name']))
    if 'data_split_file' not in dataset_metadata:
        raise ValueError(f"Dataset {cl_dataset.name} (ID: {cl_dataset.id} does not have a CV split.")
    cv_file_path = str(os.path.join(base_path, 'cv_split', os.path.basename(dataset_metadata['data_split_file'])))
    return images_path, labels_path, cv_file_path


def setup_clearml_training(config: Dict, config_path: str):
    """
    Prep clearML IDs

    :param config:          Full config
    :param config_path:     Path to full config
    :return:
    """
    task_prev = None
    if running_remotely():
        task = Task.current_task()
    else:
        task = Task.init(project_name='RSNABinary',
                         task_name=config_path.split(os.sep)[-1],
                         reuse_last_task_id=False,
                         auto_connect_frameworks={'matplotlib': False})
        if config['continue_training_from_clearml'] is not None:
            task_prev = Task.get_task(task_id=config['continue_training_from_clearml'])
            print(f'Use ckpts from task {config['continue_training_from_clearml']}')
    config = task.connect(config)
    task.add_tags(config['clearml_tags'])
    return task, task_prev


def get_ckpt_path(config: Dict, curr_subfolder_cv: str, cv_idx: int, task_prev: Task) -> str:
    """
    Prep the ckpt path based on whether a ClearML base task is used.

    :param config:              Full config
    :param curr_subfolder_cv:   Curr CV folder for putting the potential ckpt
    :param cv_idx:              Subfolder name
    :param task_prev:           ClearML task to get the previous model from
    :return:
    """
    if config['continue_training'] is not None and os.path.isfile(os.path.join(curr_subfolder_cv, mc.CKPT_NAME)):
        ckpt_path = os.path.join(curr_subfolder_cv, mc.CKPT_NAME)
    elif task_prev is not None:
        print('No local ckpt found, getting ckpt from ClearML')
        ckpt_path = os.path.join(curr_subfolder_cv, mc.CKPT_NAME)
        ckpt_path_tmp = task_prev.artifacts[f"model-CV{cv_idx + 1}-latest"].get_local_copy()
        shutil.copy(ckpt_path_tmp, ckpt_path)
    else:
        ckpt_path = None
    return ckpt_path


def setup_local_folders(config: Dict, cv_file_path: str, config_path: str, clearml: bool, task: Task) -> str:
    """
    Set up local folders

    :param config:          Full config
    :param cv_file_path:
    :param config_path:
    :param clearml:         Using clearml
    :param task:            ClearML task
    :return:
    """
    curr_subfolder = os.path.join(os.path.normpath(config['base_path']), 'experiments', config_path.replace('\\', '/').split('/')[-1])
    if config['continue_training'] is not None:
        curr_subfolder += config['continue_training']
    else:
        curr_subfolder += datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    # Copy CV file to folder and clearml for later use in evaluation
    os.makedirs(curr_subfolder, exist_ok=True)
    shutil.copy(os.path.normpath(config_path), os.path.join(curr_subfolder, 'config.yaml'))
    shutil.copy(cv_file_path, os.path.join(curr_subfolder, 'cv_split_file.json'))
    if clearml:
        # Write to disk for easier access
        with open(os.path.join(curr_subfolder, 'clearml_id.txt'), 'w', encoding='utf-8') as file:
            file.write(f"{task.task_id}")
    if task is not None:
        task.upload_artifact('cv_split_file', os.path.normpath(cv_file_path))
    return curr_subfolder


def get_callbacks(config: Dict, prefix: str, curr_subfolder_cv: str, iter_per_epoch: int) -> List[Callback]:
    callback_list = list()
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor=config['val_best_metric'] + prefix,
        save_last=True,
        save_top_k=1,
        dirpath=curr_subfolder_cv,
        filename='train-{epoch:02d}-{' + config['val_best_metric'] + ':.4f}',
        mode=config['val_best_target'],
        auto_insert_metric_name=True,
        every_n_epochs=config['ckpt_every_n_epochs'],
    )
    callback_list.append(checkpoint_callback)
    callback_list.append(MBPlotCallback(curr_subfolder_cv, config))
    callback_list.append(LearningRateMonitor(logging_interval='epoch'))
    callback_list.append(RichProgressBar(refresh_rate=iter_per_epoch // 5))
    return callback_list
