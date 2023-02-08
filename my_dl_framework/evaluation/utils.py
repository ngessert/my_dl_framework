import os
import shutil

import torch
import numpy as np
from typing import Dict, List, Tuple

from clearml import Task
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score, log_loss
import plotly.graph_objects as go
import plotly.figure_factory as ff

import my_dl_framework.constants as mc


def get_roc_curve(class_names: List[str], predictions: np.ndarray, targets: np.ndarray) -> Tuple[go.Figure, List]:
    """
    Returns a plotly ROC curve and AUC scores
    :param class_names:         Class names, list
    :param predictions:         2D array of predictions
    :param targets:             1D array of targets (ints)
    :return:                    plotly figure, list of auc scores
    """
    # Create an empty figure, and iteratively add new lines
    # every time we compute a new class
    fig = go.Figure()
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    auc_scores = list()
    for i in range(predictions.shape[1]):
        y_score = predictions[:, i]
        y_true = np.array(targets == i).astype(int)

        fpr, tpr, _ = roc_curve(y_true, y_score)
        try:
            auc_score = roc_auc_score(y_true, y_score)
        except ValueError as e:
            auc_score = 0.0
            print(f"Warning: no AUC caluclated, error: {e}")
        auc_scores.append(auc_score)
        name = f"{class_names[i]} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain='domain'),
        width=700, height=500
    )
    return fig, auc_scores


def get_confusion_matrix(conf_matrix: np.ndarray, class_names: List[str]) -> go.Figure:
    """
    Create confusion matrix plotly pot
    :param conf_matrix:         Confusion matrix as numpy array
    :param class_names:         Class names
    :return:
    """
    # invert z idx values
    z = conf_matrix[::-1]

    x = class_names
    y = x[::-1].copy()  # invert idx values of x

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis', showscale=True)

    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>',
                      xaxis=dict(title='Predicted'),
                      yaxis=dict(title='Target')
                      )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Predicted value",
                            ))
    return fig


def calculate_metrics(predictions: np.ndarray, 
                      targets: np.ndarray,
                      class_names: List[str]) -> Tuple[Dict, Dict]:
    """ Calculate classification metrics from predictions (BxC) 
        and targets (B)
    """
    metrics = dict()
    plots = dict()
    try:
        log_loss_value = log_loss(targets, predictions)
    except ValueError as e:
        log_loss_value = 100
        print(f"Warning: no log loss caluclated, error: {e}")
    log_loss_value = np.repeat(log_loss_value, len(class_names) +1)
    accuracy = np.zeros([len(class_names) + 1])
    accuracy[:-1] = np.mean(np.equal(np.argmax(predictions, 1), targets))
    accuracy[-1] = np.mean(accuracy[:-1])
    conf_matrix = confusion_matrix(targets, np.argmax(predictions, 1), labels=np.arange(len(class_names)))
    plots["confusion_matrix"] = get_confusion_matrix(conf_matrix=conf_matrix, class_names=class_names)
    # Sensitivity / Specificity
    sensitivity = np.zeros([len(class_names) + 1])
    specificity = np.zeros([len(class_names) + 1])
    precision = np.zeros([len(class_names) + 1])
    for k in range(len(class_names)):
        sensitivity[k] = conf_matrix[k, k] / (np.sum(conf_matrix[k, :]))
        true_negative = np.delete(conf_matrix, [k], 0)
        true_negative = np.delete(true_negative, [k], 1)
        true_negative = np.sum(true_negative)
        false_positive = np.delete(conf_matrix, [k], 0)
        false_positive = np.sum(false_positive[:, k])
        true_positive = conf_matrix[k, k]
        specificity[k] = true_negative / (true_negative + false_positive)
        precision[k] = true_positive / (true_positive + false_positive)
    # Add average
    sensitivity[-1] = np.mean(sensitivity[:-1])
    specificity[-1] = np.mean(specificity[:-1])
    precision[-1] = np.mean(precision[:-1])
    # F1 score
    f1 = np.zeros([len(class_names) + 1])
    f1[:-1] = f1_score(np.argmax(predictions, 1), targets, average='weighted')
    f1[-1] = np.mean(f1[:-1])
    # AUC
    auc_scores = np.zeros([len(class_names) + 1])
    fig, auc_scores[:-1] = get_roc_curve(class_names=class_names, predictions=predictions, targets=targets)
    auc_scores[-1] = np.mean(auc_scores[:-1])
    plots["roc_curve"] = fig
    metrics["log_loss_value"] = log_loss_value
    metrics["accuracy"] = accuracy
    metrics["f1_score"] = f1
    metrics["sensitivity"] = sensitivity
    metrics["specificity"] = specificity
    metrics["precision"] = precision
    metrics["auc_score"] = auc_scores
    return metrics, plots


def aggregate_pred_and_tar(config: Dict,
                           all_predictions: Dict,
                           all_targets: Dict,
                           class_names: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Helper function to aggregate predictions and targets into arrays
    :param config:              Dict with config
    :param all_predictions:     Dict of predictions
    :param all_targets:         Dict of targets
    :param class_names:         List of class names
    :return:
    """
    all_predictions_arr = np.zeros([len(all_predictions), len(class_names)])
    all_targets_arr = np.zeros([len(all_predictions)], dtype=np.int32)
    for idx, key in enumerate(all_predictions):
        if config["test_aug_ensemble_mode"] == "mean":
            all_predictions_arr[idx, :] = np.mean(np.array(all_predictions[key]), axis=0)
        else:
            all_predictions_arr[idx, :] = all_predictions_arr[key][0]
        all_targets_arr[idx] = all_targets[key][0]
    return all_predictions_arr, all_targets_arr


def validate_model_classification(model: torch.nn.Module,
                                  dataloader: DataLoader,
                                  config: Dict,
                                  max_num_batches: int = None,
                                  use_cleaml: bool = True) -> Tuple[Dict, Dict, np.ndarray, np.ndarray]:
    """
    Validates a model using data from a dataloader
    :param model:           Torch model
    :param dataloader:      Dataloader
    :param config:          Dict with config
    :param max_num_batches: Maximum number of batches to use
    :param use_cleaml:      Whether clearml is used
    :return:                Metrics, plots, predictions, targets
    """
    model.eval()
    class_names = config["class_names"]
    # Get predictions
    all_predictions = dict()
    all_targets = dict()
    for batch_idx, (indices, images, targets) in tqdm(enumerate(dataloader), disable=use_cleaml):
        if max_num_batches is not None and max_num_batches == batch_idx:
            break
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            outputs = F.softmax(model(images), dim=1)
        outputs = outputs.data.cpu().numpy()
        targets = targets.data.cpu().numpy()
        for idx, prediction, target in zip(indices, outputs, targets):
            if idx not in all_predictions:
                all_predictions[idx] = list()
            if idx not in all_targets:
                all_targets[idx] = list()
            all_predictions[idx].append(prediction)
            all_targets[idx].append(target)
    # Put into proper arrays
    all_predictions_arr, all_targets_arr = aggregate_pred_and_tar(config, all_predictions, all_targets, class_names)
    metrics, plots = calculate_metrics(all_predictions_arr, all_targets_arr, class_names)
    return metrics, plots, all_predictions_arr, all_targets_arr


def check_if_artifacts_exist(prev_task: Task, config: Dict, clearml_id: str):
    """
    Check if the required artifacts are there
    :param prev_task:       ClearML task to get artifacts from
    :param config:          Config dict
    :param clearml_id:      ClearML id of task
    :return:
    """
    # Check if required artifact exists
    if "pred arr _CV_avg" not in prev_task.artifacts and config["ensemble_type"] == "predict":
        raise ValueError(f"ClearML task {clearml_id} does not have a prediction-style predictions file. Did you "
                         f"specify a successful evaluation task?")
    if "tar arr _CV_avg" not in prev_task.artifacts and config["ensemble_type"] == "predict":
        raise ValueError(f"ClearML task {clearml_id} does not have a prediction-style targets file. Did you "
                         f"specify a successful evaluation task?")
    if "pred arr _CV_cat" not in prev_task.artifacts and config["ensemble_type"] == "eval":
        raise ValueError(f"ClearML task {clearml_id} does not have a CV-eval-style predictions file. Did you "
                         f"specify a successful evaluation task?")
    if "tar arr _CV_cat" not in prev_task.artifacts and config["ensemble_type"] == "eval":
        raise ValueError(f"ClearML task {clearml_id} does not have a CV-eval-style targets file. Did you specify "
                         f"a successful evaluation task?")
    if "training_config" not in prev_task.artifacts:
        raise ValueError(f"ClearML task {clearml_id} does not training config file. Did you specify "
                         f"a successful evaluation task?")


def add_experiments_from_clearml(config: Dict, all_eval_folders: List[str]) -> List[str]:
    """
    Get experiments from clearml
    :param config:              Config dict
    :param all_eval_folders:    List of folders with experiments
    :return:
    """
    for clearml_id in config['clearml_ids']:
        print(f"Retrieving artifacts from ClearML ID {clearml_id}")
        prev_task = Task.get_task(task_id=clearml_id)
        check_if_artifacts_exist(prev_task, config, clearml_id)
        curr_folder = os.path.join(os.path.normpath(config['save_path']), clearml_id)
        os.makedirs(curr_folder, exist_ok=True)
        path_to_file = prev_task.artifacts["training_config"].get_local_copy()
        shutil.copy(path_to_file, os.path.join(curr_folder, mc.TRAIN_CFG_NAME))
        if config["ensemble_type"] == "eval":
            if not os.path.exists(os.path.join(curr_folder, mc.PRED_FILE_NAME)):
                path_to_file = prev_task.artifacts["pred arr _CV_cat"].get_local_copy()
                shutil.copy(path_to_file, os.path.join(curr_folder, mc.PRED_FILE_NAME))
            if not os.path.exists(os.path.join(curr_folder, mc.TAR_FILE_NAME)):
                path_to_file = prev_task.artifacts["tar arr _CV_cat"].get_local_copy()
                shutil.copy(path_to_file, os.path.join(curr_folder, mc.TAR_FILE_NAME))
        elif config["ensemble_type"] == "predict":
            if not os.path.exists(os.path.join(curr_folder, mc.PRED_FILE_NAME)):
                path_to_file = prev_task.artifacts["pred arr _CV_avg"].get_local_copy()
                shutil.copy(path_to_file, os.path.join(curr_folder, mc.PRED_FILE_NAME))
            if not os.path.exists(os.path.join(curr_folder, mc.TAR_FILE_NAME)):
                path_to_file = prev_task.artifacts["tar arr _CV_avg"].get_local_copy()
                shutil.copy(path_to_file, os.path.join(curr_folder, mc.TAR_FILE_NAME))
        # Add to normal folder list
        all_eval_folders.append(curr_folder)
    return all_eval_folders


def gather_predictions(all_eval_folders: List[str]) -> Tuple[List, List, Dict]:
    """
    Gather predicitions from all folders in to lists
    :param all_eval_folders:        List with folders
    :return:
    """
    pred_list = list()
    tar_list = list()
    training_config = None
    for curr_subfolder in all_eval_folders:
        if not os.path.isfile(os.path.join(curr_subfolder, mc.PRED_FILE_NAME)):
            raise ValueError(f"Could not find predictions_all.npy in folder {curr_subfolder}")
        if not os.path.isfile(os.path.join(curr_subfolder, mc.TAR_FILE_NAME)):
            raise ValueError(f"Could not find targets_all.npy in folder {curr_subfolder}")
        if training_config is None and os.path.isfile(os.path.join(curr_subfolder, mc.TRAIN_CFG_NAME)):
            with open(os.path.join(curr_subfolder, mc.TRAIN_CFG_NAME), encoding="utf-8") as file:
                training_config = yaml.safe_load(file)
        pred = np.load(os.path.join(curr_subfolder, mc.PRED_FILE_NAME))
        pred_list.append(pred)
        tar_list.append(np.load(os.path.join(curr_subfolder, mc.TAR_FILE_NAME)))
    return pred_list, tar_list, training_config
