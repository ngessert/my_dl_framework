"""CV split generation script for rsna challenge.
Usage:
    define_cv_split_rsnachallenge.py [-h] --num_cv=<num_cv> --save_path=<save_path>

Options:
    -h --help                   Display help
    --num_cv=<num_cv>           Number of cv splits
    --save_path=<save_path>     file.json containing the defined splits
Example:
    python my_dl_framework\\data\\define_cv_split_rsnachallenge.py --num_cv=3 --save_path=C:\\sources\\my_dl_framework\\cv_splits\\cv_split_3fold.json
"""
import argparse

import pandas as pd
import os
import json


def main(num_cv: int,
         save_path: str):
    # labels = pd.read_csv(os.path.join(r"C:\data\RSNA_challenge", "stage_2_train_labels.csv"))
    labels = pd.read_csv(os.path.join(r"C:\sources\my_dl_framework\tests\test_data\rsna_binary", "stage_2_train_labels_test.csv"))
    all_cases = labels['patientId'].tolist()
    # For balancing
    all_targets = labels['Target'].tolist()
    cases_per_class = dict()
    for target, case in zip(all_targets, all_cases):
        if target not in cases_per_class:
            cases_per_class[target] = list()
        cases_per_class[target].append(case)
    cv_cases_dict = dict()
    cv_cases_dict["validation_splits"] = [[] for _ in range(num_cv)]
    cv_cases_dict["training_splits"] = [[] for _ in range(num_cv)]
    for target in cases_per_class:
        idx = 0
        while idx < len(cases_per_class[target]):
            for cv_split in range(num_cv):
                if idx < len(cases_per_class[target]):
                    cv_cases_dict["validation_splits"][cv_split].append(cases_per_class[target][idx])
                    idx += 1
    # Create traing splits from val splits
    for cv_split in range(num_cv):
        cv_cases_dict["training_splits"][cv_split] = [case for case in all_cases if case not in cv_cases_dict["validation_splits"][cv_split]]
    # Save
    with open(os.path.normpath(save_path), "w") as f:
        json.dump(cv_cases_dict, f, indent=4)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-n', '--num_cv', type=int, help='Number of CV splits', required=True)
    argparser.add_argument('-s', '--save_path', type=str, help='Save dir', required=True)
    args = argparser.parse_args()
    print(f'Args: {args}')
    main(num_cv=args.num_cv,
         save_path=args.save_path)
