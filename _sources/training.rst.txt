================
Training
================

This section provides a detailed overview on the options for training.

First, you need to set up the data and a base config with the correct structure.

Prepare Data and Config
=======

Download the data (e.g. the `RSNA Challenge Data <https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data>`_) and place it an directory, e.g., named `RSNA_challenge`.
This directory should contain a folder with the images (e.g. named `training_images`) and the CSV with labels (e.g. named `stage_2_train_labels.csv`).

Optionally, you can create a cross-validation (CV) split file (or you the default one for the RSNA challenge `cv_splits\cv_split_3fold.json`)::

    python my_dl_framework\\data\\define_cv_split_rsnachallenge.py --num_cv=3 --save_path=C:\\cv_split_3fold.json

We will treat any training as a CV training. Even if you just want to train one model with a simple train/val, split we do that by defining CV splits in a JSON file.
This ensures that data splits are guaranteed to be reproducible.

Next, prepare a training config by copying the config `configs/test_config.yaml`. Update the location of your training data and CV-split file::

    base_path: "C:\\data\\RSNA_challenge"
    csv_name: "stage_2_train_labels.csv"
    training_image_dir: "training_images"
    data_split_file: "C:\\cv_split_3fold.json"


Execute the Training
=======

If you want to use ClearML for experiment tracking, set up an account at `https://clear.ml <https://clear.ml>`_.
Log in and generate App credentials by going to Settings -> Workspaces. Copy the generated credentials into the `clearml.conf` file in the repo root.
Then, copy the clearml.conf file to your machines home directory.

You can run the training locally with::

    python my_dl_framework\\training\\train_pl.py --config=C:\\sources\\my_dl_framework\\configs\\test_config.yaml -cl clearml


Config Options
=======

The following config options are available: TODO

Remote Training and ClearML Workers
=======

You can also use ClearML queues and workers to schedule multiple jobs. This can also make sense if just work
with a single, local computer for running your trainings.

TODO: write description