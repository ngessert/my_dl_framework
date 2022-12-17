import argparse


def run_evaluation(args):
    pass
    # TODO: load config from path or Clearml

    # TODO: initalize cross-CV dicts etc.

    # TODO: loop over CV splits

    # TODO: Save metrics & push to ClearML


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-p', '--path', type=str, help='Path to completed training', required=False)
    argparser.add_argument('-clid', '--clearml_id', type=str, help='ClearML training job ID', required=False)
    argparser.add_argument('-cl', '--clearml', type=bool, default=False, help='Whether to use clearml', required=False)
    argparser.add_argument('-r', '--remote', type=bool, default=False, help='Whether remote execution is done', required=False)
    args = argparser.parse_args()
    if args.path is None and args.clearml_id is None:
        raise ValueError("Specify path or clearmlid to use for evaluation.")
    if args.path is not None and args.clearml_id is not None:
        raise ValueError("Set eithe path or clearml_id, not both.")
    print(f'Args: {args}')
    run_evaluation(args)