import os
import json
import argparse
import glob 
from configparser import ConfigParser

from kfp.v2 import compiler
import google.cloud.aiplatform as vertex_ai

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


def execute_train_pipeline(
        project: str, 
        region: str, 
        service_account: str, 
        storage_bucket: str,
        pipeline_root: str,
        cfg: ConfigParser
):
    """
    Submits training pipeline job to Vertex AI
    """

    from pipeline.pipeline_train import build_training_pipeline

    vertex_ai.init(
        project=project,
        location=region,
        staging_bucket=storage_bucket,
        service_account=service_account
    )

    # Compile Pipeline
    compiler.Compiler().compile(
        pipeline_func=build_training_pipeline(cfg=cfg), 
        package_path=pipeline_package_path
    )
    
    # Initialize pipeline job
    job = vertex_ai.PipelineJob(
        display_name=pipeline_display_name, 
        template_path=pipeline_package_path,
        pipeline_root=pipeline_root,
        parameter_values={
            'project': project,
            'region': region,
        }
    )
    
    # Run pipeline job
    job.submit(service_account=service_account)


def execute_inference_pipeline(
        project: str, 
        region: str, 
        storage_bucket: str, 
        pipeline_root: str, 
        service_account: str, 
        cfg: ConfigParser
):
    """
    Submits inference pipeline job to Vertex AI
    """
    pass


def main():
    """"""
    args = get_args()

    if args.command == "train-model":
        execute_train_pipeline(
            project=args.project,
            region=args.region,
            storage_bucket=args.storage_bucket,
            pipeline_root=args.pipeline_root,
        )

    elif args.command == 'deploy-model':
        execute_inference_pipeline(
            

        )

    else:
        raise ValueError(f'Invalid command-line argument : {args.command}')
    

if __name__ == "__main__":
    main()

