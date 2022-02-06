"""CV split generation script for rsna challenge.
Usage:
    define_cv_split_rsnachallenge.py [-h] --num_cv=<num_cv> --save_path=<save_path>

Options:
    -h --help                   Display help
    --num_cv=<num_cv>           Number of cv splits
    --save_path=<save_path>     file.json containing the defined splits
Example:
    python my_dl_framework\\data_handling\\define_cv_split_rsnachallenge.py --num_cv=3 --save_path=C:\\sources\\my_dl_framework\\cv_splits\\cv_split_3fold.json
"""

from docopt import docopt
import pandas as pd
import numpy as np
import os
import json


def main():
    args = docopt(__doc__)
    num_cv = int(args["--num_cv"])
    labels = pd.read_csv(os.path.join(r"C:\data\RSNA_challenge", "stage_2_train_labels.csv"))
    all_cases = labels['patientId'].tolist()
    # For balancing
    all_targets = labels['Target'].tolist()
    num_classes = len(np.unique(all_targets))
    cases_per_class = dict()
    for target, case in zip(all_targets, all_cases):
        if target not in cases_per_class:
            cases_per_class[target] = list()
        cases_per_class[target].append(case)
    cv_cases_dict = dict()
    cv_cases_dict["validation_splits"] = [[] for _ in range(num_cv)]
    cv_cases_dict["training_splits"] = [[] for _ in range(num_cv)]
    idx = 0
    for target in cases_per_class:
        while idx < len(cases_per_class[target]):
            for cv_split in range(num_cv):
                if idx < len(cases_per_class[target]):
                    cv_cases_dict["validation_splits"][cv_split].append(cases_per_class[target][idx])
                    idx += 1
    # Create traing splits from val splits
    for cv_split in range(num_cv):
        cv_cases_dict["training_splits"][cv_split] = [case for case in all_cases if case not in cv_cases_dict["validation_splits"][cv_split]]
    # Save
    with open(os.path.normpath(args["--save_path"]), "w") as f:
        json.dump(cv_cases_dict, f, indent=4)


if __name__ == "__main__":
    main()
