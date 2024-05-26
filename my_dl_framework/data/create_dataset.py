"""
Creates a ClearML dataset to be used for training. Can include preprocessing.
This dataset can be consumed by a trained or evaluation script.
"""
import argparse
import os
from glob import glob

import pandas as pd
import yaml
from clearml import Dataset

from my_dl_framework.data.get_dataset import get_dataset_stats


def create_dataset(config_path: str) -> str:
    """
    Creates a ClearML data set for logging purposes.
    :param config_path:         Config that contains a data section
    :return:
    """
    with open(config_path, encoding='utf-8') as file:
        data_config = yaml.safe_load(file)['data']
        print(f'Using data config {config_path}')
    # We can start at a folder path or a ClearML dataset
    parent_dataset_id = None
    src_labels = None
    src_cv_file_path = None
    if data_config['images_dir'] is not None:
        src_images = glob(os.path.normpath(data_config['images_dir']) + '/*')
        if data_config['labels_dir']:
            src_labels = glob(os.path.normpath(data_config['labels_dir']) + '/*')
        if data_config['data_split_file']:
            src_cv_file_path = glob(os.path.normpath(os.path.dirname(data_config['data_split_file'])) + '/*')
    elif data_config['cl_base_dataset_id'] is not None:
        base_dataset = Dataset.get(dataset_id=data_config['cl_base_dataset_id'])
        src_data_path = base_dataset.get_local_copy(use_soft_links=True)
        dataset_metadata = base_dataset.get_metadata().to_dict()
        src_images = os.path.join(src_data_path, dataset_metadata['images_dir'])
        if 'labels_dir' in dataset_metadata:
            src_labels = os.path.join(src_data_path, dataset_metadata['labels_dir'])
        if 'data_split_file' in dataset_metadata:
            src_cv_file_path = os.path.join(src_data_path, os.path.basename(dataset_metadata['data_split_file']))
        parent_dataset_id = base_dataset.id
    else:
        raise ValueError(f"No data source defined in config {config_path}")
    # Apply preprocessing TODO

    # Finalize dataset
    new_dataset = Dataset.create(dataset_name=data_config['new_dataset_name'],
                                 dataset_project=data_config['dataset_project'],
                                 parent_datasets=parent_dataset_id,
                                 output_uri=data_config['dataset_output_uri'],
                                 description=data_config.get('dataset_description'))
    dataset_metadata = {'images_dir': [os.path.basename(data_config['images_dir'])]}
    if src_cv_file_path:
        dataset_metadata['data_split_file'] = [data_config['data_split_file']]
        new_dataset.add_external_files(src_cv_file_path, wildcard='*.json', dataset_path='cv_split')
    new_dataset.add_external_files(src_images, dataset_path=os.path.basename(data_config['images_dir']))
    if src_labels:
        dataset_metadata['labels_dir'] = [os.path.basename(data_config['labels_dir'])]
        new_dataset.add_external_files(src_labels, dataset_path=os.path.basename(data_config['labels_dir']))
        if 'csv_name' in data_config:
            dataset_metadata['csv_name'] = [data_config['csv_name']]
    new_dataset.set_metadata(pd.DataFrame.from_dict(dataset_metadata).T)
    # Add data statistics
    dataset_logger = new_dataset.get_logger()
    for fig_title, fig in get_dataset_stats({'label_csv': os.path.join(data_config['labels_dir'], data_config['csv_name'])}, data_config['dataset_type']).items():
        dataset_logger.report_plotly(fig_title, fig_title, fig, 0)

    new_dataset.finalize(auto_upload=True)
    return new_dataset.id


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-c', '--config', type=str, help='Config path', required=True)
    args = argparser.parse_args()
    print(f'Args: {args}')
    create_dataset(args.config)
