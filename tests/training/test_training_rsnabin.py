"""
Some basic tests
"""
import os
from glob import glob
from my_dl_framework.training.train_pl import run_training


def test_train_script():
    run_training(config_path=os.path.join("tests", "test_configs", "test_config.yaml"),
                 clearml=False,
                 num_gpus=None,
                 multi_gpu_strat=None,
                 remote=False,
                 fast_dev_run=False)
    found_experiment = False
    for folder in glob(os.path.join("tests", "test_data", "rsna_binary", "experiments", "*")):
        if "test_config" in folder:
            assert os.path.exists(os.path.join(folder, "CV_1", "last.ckpt")), "CV1 ckpt not found"
            assert os.path.exists(os.path.join(folder, "CV_1", "example_batches", "batch_0.png")), "Example batch plot not found"
            assert os.path.exists(os.path.join(folder, "CV_2", "last.ckpt")), "CV2 ckpt not found"
            found_experiment = True
    assert found_experiment, "Experiment run not found"
    # os.remove(os.path.join("tests", "test_data", "rsna_binary", "experiments"))
