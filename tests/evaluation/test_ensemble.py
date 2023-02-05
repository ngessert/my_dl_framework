"""
Some basic tests
"""
import os
from glob import glob

from my_dl_framework.evaluation.ensemble import run_ensemble


def test_eval_and_pred_script():
    run_ensemble(config_path=os.path.join("tests", "test_configs", "test_ensemble.yaml"),
                 clearml=False,
                 remote=False)
    found_experiment = False
    for folder in glob(os.path.join("tests", "test_data", "rsna_binary", "exp_for_val", "*")):
        if "pred_and_eval" in folder:
            assert os.path.exists(os.path.join(folder, "predictions_all.npy")), "predictions_all.npy not found"
            assert os.path.exists(os.path.join(folder, "targets_all.npy")), "targets_all.npy not found"
            found_experiment = True
    assert found_experiment, "Ensemble experiment run not found"
    # os.remove(os.path.join("tests", "test_data", "rsna_binary", "experiments"))
