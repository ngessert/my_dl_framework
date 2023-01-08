"""Training script for remote ClearML execution.
Usage:
    train_remote.py --config=<config>

Options:
    --config=<config>    config name to run.
Example:
    python my_dl_framework\\training\\train_remote.py --config=C:\\sources\\my_dl_framework\\configs\\test_config.yaml
"""
import os
import argparse
from clearml import Task, TaskTypes


def train_remote(cmd_args):
    """ Execute remote job by enqueing it to a clearml queue
    """
    # Add arguments
    cmd_args["clearml"] = True
    cmd_args["remote"] = True
    # Define tas
    task = Task.create(project_name='RSNABinary',
                       task_name=cmd_args.config.split(os.sep)[-1],
                       task_type=TaskTypes.training,
                       repo="https://github.com/ngessert/my_dl_framework",
                       branch=cmd_args.branch,
                       script="./my_dl_framework/training/train_pl.py",
                       argparse_args=[(key, value) for key, value in vars(cmd_args).items()],
                     )
    Task.enqueue(task=task, queue_name="default")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', type=str, help='Config path', required=True)
    argparser.add_argument('-b', '--branch', type=str, help='Branch', required=False, default="develop")
    args = argparser.parse_args()
    print(f'Args {args}')
    train_remote(args)
