Evaluation and Ensemlbing
==========

Afer training, you can run evaluations and prediction on the training data or new data. You can either use a **local** folder
with trained model checkpoints or a ClearML **job id**. For the latter case, the model checkpoints, configs, etc. are downloaded from
ClearML.

Evaluation with local folder::

    python my_dl_framework\evaluation\eval_and_predict.py --folder=experiments\training1--image_path=C:\\training_images -cl

Also, you can either run an evaluation in **CV mode** (on the training dataset) or **prediction mode** on new data.
For the former, each CV model is applied to its respective data subset and the predictions are concatenated together for
evaluation across CV splits. For the latter, each CV model is applied to all data and predictions are averaged across folds.

For **CV mode** add the flag `--cv_eval`.

The **prediction mode** is meant for new, unlabeled data, however, you can also provide CSV file in case labels are available: `--label_csv_path=label_file.csv`.

There are multiple **test-time augmentation** options that can be arbitrarily combined. E.g., use flipping augmentation with `--tta_flip_horz` and/or `--tta_flip_vert`.

For each evaluation run, a new subfolder is created in the original training folder, containing the predictions and logs. Also,
metrics and prediction files are logged in ClearML. Multiple predictions can be used for ensembling.

In addition, there are more options available: TODO