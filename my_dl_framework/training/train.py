"""Training script.
Usage:
    train.py --config=<config> [-htvp]

Options:
    -h --help            show this message and exit.
    --config=<config>    config name to run.
    -t --training        run training.
    -v --validate        run validation.
    -p --predict         run prediction on new data.
"""

from docopt import docopt
import json
import os
import numpy as np
import torch
from tqdm import tqdm
from glob import glob
from torch.utils.data import DataLoader
from utils import get_dataset, get_model, get_lossfunction, get_optimizer, get_lr_scheduler


def main():
    args = docopt(__doc__)
    # Import config
    with open(os.path.join("../../configs", args["--config"])) as f:
        config = json.load(f)
    # Run Training
    if args["--training"]:
        # Setup CV
        with open(os.path.join(config['base_path'], config["data_split_file"])) as f:
            data_split = json.load(f)
        training_subsets = data_split["training_splits"]
        validation_subsets = data_split["validation_splits"]
        for idx, (subset_train, subset_val) in enumerate(zip(training_subsets, validation_subsets)):
            print(f'Starting fold number {idx+1}/{len(training_subsets)}')
            curr_subfolder = os.path.join(config['base_path'], "experiments", args["--config"], "CV_" + str(idx+1))
            # Data
            dataset_train = get_dataset(config=config, image_dir=config["training_image_dir"], subset=subset_train,
                                        is_training=True)
            print(f'Size training dataset {len(dataset_train)}')
            dataset_val = get_dataset(config=config, image_dir=config["training_image_dir"], subset=subset_val,
                                      is_training=False)
            print(f'Size validation dataset {len(dataset_val)}')

            def worker_init(worker_id):
                return np.random.seed(np.random.get_state()[1][0] + worker_id)

            dataloader_train = DataLoader(dataset=dataset_train, batch_size=config["batch_size"], shuffle=True,
                                          num_workers=8, pin_memory=True,
                                          worker_init_fn=None)
            dataloader_val = DataLoader(dataset=dataset_val, batch_size=config["batch_size"], shuffle=False,
                                        num_workers=8, pin_memory=True)
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
                    metric_dict = dict()
                else:
                    start_epoch = 0
                    metric_dict = dict()
            else:
                metric_dict = dict()
                start_epoch = 0
                os.makedirs(curr_subfolder, exist_ok=False)
            # Training
            for epoch in range(start_epoch, config['num_epochs']):
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
                    if "train_loss" not in metric_dict:
                        metric_dict["train_loss"] = np.array([[epoch * len(dataloader_train) + batch_idx,
                                                               loss.detach().cpu().numpy()]])
                    else:
                        metric_dict["train_loss"] = np.concatenate((metric_dict["train_loss"], np.array(
                            [[epoch*len(dataloader_train) + batch_idx, loss.detach().cpu().numpy()]])), axis=0)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                print(f'Fold {idx+1}/{len(training_subsets)} Epoch {epoch}/{config["num_epochs"]} completed. '
                      f'Last train loss: {loss.detach().cpu().numpy()}')
                # Save model/optimizer
                state_opt = {'state_dict': optimizer.state_dict()}
                torch.save(state_opt, os.path.join(curr_subfolder, "optimizer_ckpt_" + str(epoch) + ".pt"))
                state_model = {'state_dict': model.state_dict()}
                torch.save(state_model, os.path.join(curr_subfolder, "model_ckpt_" + str(epoch) + ".pt"))
                # Remove previous one
                if os.path.exists(os.path.join(curr_subfolder, "optimizer_ckpt_" + str(epoch-1) + ".pt")):
                    os.remove(os.path.join(curr_subfolder, "optimizer_ckpt_" + str(epoch-1) + ".pt"))
                if os.path.exists(os.path.join(curr_subfolder, "model_ckpt_" + str(epoch-1) + ".pt")):
                    os.remove(os.path.join(curr_subfolder, "model_ckpt_" + str(epoch-1) + ".pt"))
                # Validate in between
                if epoch % config['validate_every_x_epochs'] == 0:
                    print(f'Validation not yet implemented')

                # Save metrics


if __name__ == "__main__":
    main()
