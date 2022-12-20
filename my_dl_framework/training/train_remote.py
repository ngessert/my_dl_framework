"""Training script for remote ClearML execution.
Usage:
    train.py --config=<config> [-htvpc]

Options:
    --config=<config>    config name to run.
    -cl --clearml         use clearml.
Example:
    python my_dl_framework\\training\\train.py --config=C:\\sources\\my_dl_framework\\configs\\test_config.yaml -tc
"""
import os
import argparse
from clearml import Task, TaskTypes


def train_remote(cmd_args):
    """ Execute remove job
    """
    # Define tas
    task = Task.create(project_name='RSNABinary',
                       task_name=cmd_args.config.split(os.sep)[-1],
                       task_type=TaskTypes.training,
                       repo="https://github.com/ngessert/my_dl_framework",
                       branch="develop",
                       script="./my_dl_framework/training/train.py",
                       argparse_args=[(key, value) for key, value in vars(cmd_args).items()],
                     )
    Task.enqueue(task=task, queue_name="default")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', type=str, help='Config path', required=True)
    argparser.add_argument('-cl', '--clearml', type=bool, default=False, help='Whether to use clearml', required=False)
    args = argparser.parse_args()
    print(f'Args {args}')
    train_remote(args)
