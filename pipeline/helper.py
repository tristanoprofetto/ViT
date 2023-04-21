import os
import json
import argparse

import glob 


def get_args():
    """
    Command line arguments for model training/deployment pipelines
    """
    parser = argparse.ArgumentParser()
    parser_train = parser.add_subparsers('train-model')
    parser_train.add_argument('--project')
    parser_train.add_argument('--region')
    parser_train.add_argument('--service_account')
    parser_train.add_argument('--storage_bucket')
    parser_train.add_argument('--pipeline_root')
    parser_train.add_argument('--pipeline_id')
    parser_train.add_argument('--dataset_name')
    parser_train.add_argument('--device')
    parser_train.add_argument('--batch_size')
    parser_train.add_argument('--num_workers')

    parser_deploy = parser.add_subparsers('deploy-model')
    parser_deploy.add_argument('--project')
    parser_deploy.add_argument('--region')
    parser_deploy.add_argument('--service_account')
    parser_deploy.add_argument('--storage_bucket')
    parser_deploy.add_argument('--pipeline_root')
    parser_deploy.add_argument('--pipeline_id')
    parser_deploy.add_argument('--machine_type')

    args = parser.parse_args()

    return args


def update_image_tag(image_tag: str):
    pass


def train_pipeline():
    pass


def deploy_pipeline():
    pass
