"""
Some basic tests
"""
import os
from glob import glob

from my_dl_framework.evaluation.eval_and_predict import run_prediction


def test_eval_and_pred_script():
    run_prediction(folder=os.path.join("tests", "test_data", "rsna_binary", "exp_for_val"),
                   clearml_id=None,
                   clearml=False,
                   remote=False,
                   eval_on_train=True,
                   cv_eval=True,
                   image_path=os.path.join("tests", "test_data", "rsna_binary", "images"),
                   limit_batches=1,
                   dont_log_splits=False,
                   label_csv_path=os.path.join("tests", "test_data", "rsna_binary", "stage_2_train_labels_test.csv")
                   )
    found_experiment = False
    for folder in glob(os.path.join("tests", "test_data", "rsna_binary", "exp_for_val", "*")):
        if "pred_and_eval" in folder:
            assert os.path.exists(os.path.join(folder, "predictions_all.npy")), "predictions_all.npy not found"
            assert os.path.exists(os.path.join(folder, "targets_all.npy")), "targets_all.npy not found"
            assert os.path.exists(os.path.join(folder, "predictions_train_val_all.npy")), "predictions_train_val_all.npy not found"
            assert os.path.exists(os.path.join(folder, "targets_train_val_all.npy")), "targets_train_val_all.npy not found"
            assert os.path.exists(os.path.join(folder, "CV_1", "predictions.npy")), "CV1 predictions.npy not found"
            assert os.path.exists(os.path.join(folder, "CV_1", "targets.npy")), "CV1 targets.npy not found"
            assert os.path.exists(os.path.join(folder, "CV_1", "predictions_train_val.npy")), "CV1 predictions_train_val.npy not found"
            assert os.path.exists(os.path.join(folder, "CV_1", "targets_train_val.npy")), "CV1 targets_train_val.npy not found"
            found_experiment = True
    assert found_experiment, "Eval experiment run not found"
    # os.remove(os.path.join("tests", "test_data", "rsna_binary", "experiments"))
