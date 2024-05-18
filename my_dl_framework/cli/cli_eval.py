"""
CLI functions for evaluation
"""

import typer
from my_dl_framework.evaluation.eval_and_predict import run_prediction


def eval(folder: str = typer.Argument(..., help='Folder path'),
         clearml_id: str = typer.Option(None, help="ClearML ID of experiment"),
         clearml: bool = typer.Option(False, help='Use clearml tracking'),
         eval_on_train: bool = typer.Option(False, help="Whether to evaluate on the training set (when doing CV evaluation)"),
         cv_eval: bool = typer.Option(False, help="Whether the CV split file should be used for evaluation"),
         image_path: str = typer.Option(None, help="Test image path for prediction"),
         limit_batches: int = typer.Option(None, help="Test with a subset of batches"),
         dont_log_splits: bool = typer.Option(False, help="Whether inidivudal CV splits should be logged or just the cat/avg"),
         label_csv_path: str = typer.Option(None, help="Optional path to a label CSV that overrides the default one (CV eval) or is used for predict mode"),
         tta_flip_horz: bool = typer.Option(False, help="TTA"),
         tta_flip_vert: bool = typer.Option(False, help="TTA"),
         tta_multi_eq_crop: int = typer.Option(None, help="TTA"),
         ):

    run_prediction(folder=folder,
                   clearml_id=clearml_id,
                   clearml=clearml,
                   eval_on_train=eval_on_train,
                   cv_eval=cv_eval,
                   image_path=image_path,
                   limit_batches=limit_batches,
                   dont_log_splits=dont_log_splits,
                   label_csv_path=label_csv_path,
                   tta_flip_horz=tta_flip_horz,
                   tta_flip_vert=tta_flip_vert,
                   tta_multi_eq_crop=tta_multi_eq_crop)
