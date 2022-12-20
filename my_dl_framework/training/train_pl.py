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
import os
from datetime import datetime
import yaml
from torch.utils.data import DataLoader

from clearml import Task
import pytorch_lightning as pl

from my_dl_framework.models.pl_wrapper import PLClassificationWrapper
from my_dl_framework.training.utils import get_dataset
from my_dl_framework.utils.pytorch_lightning.clearml_logger import PLClearML

def run_training(cmd_args):
    """ Run pytorch lightning training
    """
    # Import config
    with open(cmd_args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)
        print(f'Using config {cmd_args.config}')
    # ClearML
    if cmd_args.clearml:
        if cmd_args.remote:
            task = Task.get_task(project_name="RSNABinary", task_name=cmd_args.config.split(os.sep)[-1])  # ?
        else:
            task = Task.init(project_name='RSNABinary',
                             task_name=cmd_args.config.split(os.sep)[-1],
                             reuse_last_task_id=False,
                             auto_connect_frameworks={'matplotlib': False}
                             )
        task.connect_configuration(config)
        pl_clearml_logger = PLClearML(task=task)
    else:
        pl_clearml_logger = None
        task = None
    # Run Training
    # Setup CV
    with open(config["data_split_file"], encoding="utf-8") as file:
        data_split = json.load(file)
    training_subsets = data_split["training_splits"]
    validation_subsets = data_split["validation_splits"]
    curr_subfolder = os.path.join(config['base_path'], "experiments", cmd_args.config.replace("\\", "/").split("/")[-1])
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
        model = PLClassificationWrapper(config=config,
                                        training_set_len=len(dataset_train))
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=None,
            save_last=True,
            save_top_k=0,
            dirpath=curr_subfolder_cv,
            filename='train-epoch{epoch:02d}',
            auto_insert_metric_name=False,
            every_n_epochs=config["ckpt_every_n_epochs"]
        )
        pl.seed_everything(42, workers=True)
        trainer = pl.Trainer(devices=cmd_args.num_gpus,
                             check_val_every_n_epoch=config["validate_every_x_epochs"],
                             default_root_dir=curr_subfolder_cv,
                             accelerator="auto",
                             max_epochs=config["num_epochs"],
                             auto_select_gpus=True if cmd_args.num_gpus is None else False,
                             strategy=cmd_args.multi_gpu_strat,
                             precision=16,
                             deterministic=True,
                             fast_dev_run=cmd_args.fast_dev_run,
                             callbacks=[checkpoint_callback],
                             logger=pl_clearml_logger,
                             )
        trainer.fit(model, dataloader_train, 
                    [dataloader_val] if dataloader_train_val is None else [dataloader_val, dataloader_train_val],
                    ckpt_path=os.path.join(curr_subfolder_cv, "last.ckpt") if config["continue_training"] else None)
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
    run_training(args)
