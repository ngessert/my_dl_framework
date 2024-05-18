"""
CLI functions for training
"""

import typer
from my_dl_framework.training.train_pl import run_training


def train(config: str = typer.Argument(..., help="Path to config"),
          num_gpus: int = typer.Option(0, help="Number of GPUs to use"),
          multi_gpu_strat: str = typer.Option("", help="MultiGPU strat, e.g. ddp"),
          clearml: bool = typer.Option(False, help='Use clearml tracking'),
          fast_dev_run: int = typer.Option(None, help="test only x batches")):

    run_training(config_path=config,
                 clearml=clearml,
                 num_gpus=num_gpus,
                 multi_gpu_strat=multi_gpu_strat,
                 fast_dev_run=fast_dev_run)
