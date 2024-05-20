"""
Typer CLI
"""
import typer

from my_dl_framework.cli.cli_eval import eval
from my_dl_framework.cli.cli_training import train


# Base
cli = typer.Typer(no_args_is_help=True)

# Training
cli.command('train', help='Entry command for training')(train)

# Evaluation
cli.command('eval', help='Entry command for evaluation')(eval)
