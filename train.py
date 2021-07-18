"""Training script.
Usage:
    train.py [-hv] [--eternal | --n_reminders=<n>] <name>
Arguments:
    name    Somebody who needs to be reminded to document their code
Options:
    -h --help            show this message and exit.
    --config=<c>         config name to run.
    -t --training        run training.
    -v --validate        run validation.
    -p --predict         run prediction on new data.
"""

from docopt import docopt
import json
import os
from utils import get_dataset


def main():
    args = docopt(__doc__, version='0.1')
    # Import config
    with open(os.path.join("configs", args["--config"])) as f:
        config = json.load(f)
    # Run Training
    if args["--training"]:
        # Data
        dataset = get_dataset(config=config)
        # Model

        # Training


if __name__ == "__main__":
    main()
