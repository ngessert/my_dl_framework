My Deep Learning Framework
==========================

|Tests Status| |Docs Status|

A Pytorch (lightning)-based framework for deep learning. The purpose of this repo is to provide tools for training and evaluating image-based deep learning models with prepared datasets.
It is focused on challenge scenarios where the goal is to train many models that are evaluated with test-time augmentation, and ensembled together.

Many components in this repo are inspired by my challenge-winning code repo that can be found here: `ISIC Challenge Code <https://github.com/ngessert/isic2019/>`_.

Features:

    *   Based on `Pytorch Lightning <https://www.pytorchlightning.ai>`_
    *   Experiment tracking & model storing with `ClearML <https://clear.ml>`_ - a powerful MLops tool
    *   A training config system with yaml files
    *   Job scheduling with ClearML workers & queues
    *   Test-time augmentation
    *   Evaluation & ensembling (also with ClearML artifacts)
    *   Supported task

        * Image classification
    *   Supported datasets

        * RSNA Pneumonia Chest X-ray Challenge (binary classification)


Planned features:

    *   Supported tasks

        * Image segmentation.
        * Image bounding box detection.
    *   Supported datasets

        * ISIC Skin Lesion Classification Challenge.


:Documentation:
    `Documentation <https://deep-echo.philips-internal.com/>`_

:Repo:
    `https://github.com/ngessert/my_dl_framework <https://github.com/ngessert/my_dl_framework>`_

Getting Started
=============================

If you plan to do local trainings only, you can clone the repo, e.g. via SSH with::

    git clone git@github.com:ngessert/my_dl_framework.git

If you plan to use **scheduling** via ClearML workers & queues, I recommend that you fork the repo as you will use
it to sync your config files.

Setup a new python environment, e.g. with virtual env or miniforge. The recommended python version is 3.9 (that's what I tested). Then, install the requirements and package with::

    pip install -r requirements.txt
    python setup.py develop



Prepare Data and Config
=============================

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

There are many other options that are explained in the `documentation <>`_.

Training
=============================

If you want to use ClearML for experiment tracking, set up an account at `https://clear.ml <https://clear.ml>`_.
Log in and generate App credentials by going to Settings -> Workspaces. Copy the generated credentials into the `clearml.conf` file in the repo root.
Then, copy the clearml.conf file to your machines home directory.

You can run the training locally with::

    python my_dl_framework\\training\\train_pl.py --config=C:\\sources\\my_dl_framework\\configs\\test_config.yaml -cl clearml

If you want to run training with ClearML workers & queues, please check the `documentation <>`_.

Evaluation
=============================

Afer training, you can an evaluation with:

Ensembling
=============================

After evaluation, you can ensemble multiple models together:


.. |Tests Status| image:: https://github.com/ngessert/my_dl_framework/actions/workflows/main.yml/badge.svg?branch=develop
.. |Docs Status| image:: https://github.com/ngessert/my_dl_framework/actions/workflows/documentation.yml/badge.svg?branch=develop