"""Training script.
Usage:
    train.py --config=<config> [-htvp]

Options:
    -h --help            show this message and exit.
    --config=<config>    config name to run.
    -t --training        run training.
    -v --validate        run validation.
    -p --predict         run prediction on new data.
Example:
    python my_dl_framework\\training\\train.py --config=C:\\sources\\my_dl_framework\\configs\\test_config.yaml -t
"""

from docopt import docopt
import json
import yaml
import os
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from utils import get_dataset, get_model, get_lossfunction, get_optimizer, get_lr_scheduler, save_optimizer_and_model


def main():
    args = docopt(__doc__)
    # Import config
    with open(args["--config"]) as f:
        config = yaml.safe_load(f)
        print(f'Using config {args["--config"]}')
    # Run Training
    if args["--training"]:
        # Setup CV
        with open(os.path.join(config['base_path'], config["data_split_file"])) as f:
            data_split = json.load(f)
        training_subsets = data_split["training_splits"]
        validation_subsets = data_split["validation_splits"]
        for idx, (subset_train, subset_val) in enumerate(zip(training_subsets, validation_subsets)):
            print(f'Starting fold number {idx+1}/{len(training_subsets)}')
            curr_subfolder = os.path.join(config['base_path'], "experiments", args["--config"].replace("\\", "/").split("/")[-1], "CV_" + str(idx+1))
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
            # Model
            model = get_model(config=config)
            model = model.cuda()
            # Loss function/optimizer
            loss_function = get_lossfunction(config=config)
            optimizer = get_optimizer(config=config, model=model)
            lr_scheduler = get_lr_scheduler(config=config, optimizer=optimizer) if config['lr_scheduler'] is not None\
                else None
            # Folder setup/perhaps load previous state
            if os.path.exists(curr_subfolder):
                # Check if checkpoint exists
                all_files = glob(os.path.join(curr_subfolder, "*"))
                max_epoch = -1
                optimizer_path = None
                for file_name in all_files:
                    if "optimizer_ckpt" in file_name:
                        curr_epoch = int(os.path.normpath(file_name).split(os.path.sep)[-1].split(".")[0])
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
                    if os.path.exists(os.path.join(curr_subfolder, "training_metrics.json")):
                        with open(os.path.join(curr_subfolder, "training_metrics.json")) as f:
                            metrics_train_all = json.load(f)
                else:
                    start_epoch = 0
                    metrics_train_all = dict()
            else:
                metrics_train_all = dict()
                start_epoch = 0
                os.makedirs(curr_subfolder, exist_ok=False)
            # Training
            for epoch in tqdm(range(start_epoch, config['num_epochs'])):
                np.random.seed(np.random.get_state()[1][0] + epoch)
                model.train()
                for batch_idx, (images, targets) in tqdm(enumerate(dataloader_train)):
                    images = images.cuda()
                    targets = targets.cuda()
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        outputs = model(images)
                        loss = loss_function(outputs, targets)
                        loss.backward()
                        optimizer.step()
                    # Track loss
                    if "train_loss" not in metrics_train_all:
                        metrics_train_all["train_loss"] = np.array([[epoch * len(dataloader_train) + batch_idx,
                                                               loss.detach().cpu().numpy()]])
                    else:
                        metrics_train_all["train_loss"] = np.concatenate((metrics_train_all["train_loss"], np.array(
                            [[epoch*len(dataloader_train) + batch_idx, loss.detach().cpu().numpy()]])), axis=0)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                print(f'Fold {idx+1}/{len(training_subsets)} Epoch {epoch}/{config["num_epochs"]} completed. '
                      f'Last train loss: {loss.detach().cpu().numpy()}')
                # Save
                save_optimizer_and_model(optimizer=optimizer, model=model, curr_subfolder=curr_subfolder,
                                         epoch=epoch, prefix="last_")
                # Save
                with open(os.path.join(curr_subfolder, "training_metrics.json")) as f:
                    json.dump(metrics_train_all, f)
                # Validate in between
                if epoch % config['validate_every_x_epochs'] == 0:
                    print(f'Validating on set of length {len(dataloader_val)}')
                    metrics = validate_model(model=model, dataloader=dataloader_val, config=config)
                    for metric in metrics:
                        print(f'{metric}: {np.mean(metrics[metric])} +- {np.std(metrics[metric])}')
                    # Save metrics
                    if os.path.exists(os.path.join(curr_subfolder, "validation_metrics.json")):
                        with open(os.path.join(curr_subfolder, "validation_metrics.json")) as f:
                            metrics_all = json.load(f)
                        metrics_all[epoch] = metrics
                        # Determine if there is a new best validation model
                        if metrics_all[metrics_all["best_epoch"]][config['val_best_metric']] < metrics[config['val_best_metric']]:
                            metrics_all["best_epoch"] = epoch
                            save_optimizer_and_model(optimizer=optimizer, model=model, curr_subfolder=curr_subfolder,
                                                     epoch=epoch, prefix="best_")
                        # Save
                        with open(os.path.join(curr_subfolder, "validation_metrics.json")) as f:
                            json.dump(metrics_all, f)
                    if config['validate_on_train_set']:
                        print(f'Validating on training set of length {len(dataloader_train_val)}')
                        metrics_train = validate_model(model=model, dataloader=dataloader_train_val, config=config)
                        for metric in metrics_train:
                            print(f'{metric}: {np.mean(metrics_train[metric])} +- {np.std(metrics_train[metric])}')
                        # Save metrics
                        if os.path.exists(os.path.join(curr_subfolder, "training_metrics.json")):
                            with open(os.path.join(curr_subfolder, "training_metrics.json")) as f:
                                metrics_train_all = json.load(f)
                            metrics_train_all[epoch] = metrics_train
                            # Save
                            with open(os.path.join(curr_subfolder, "training_metrics.json")) as f:
                                json.dump(metrics_train_all, f)


if __name__ == "__main__":
    main()
