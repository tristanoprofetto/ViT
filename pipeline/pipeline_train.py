import os
from configparser import ConfigParser

from kfp.dsl import components, pipeline
from kfp.v2.dsl import component
from google_cloud_pipeline_components.v1.custom_job import create_custom_training_job_from_component


def build_training_pipeline(cfg: ConfigParser):

    data_prepare_task = components.load_component_from_file('./component_configs/data_preperation_component.yaml')
    model_training_task = components.load_component_from_file('./component_configs/model_training_component.yaml')

    custom_training_job_op = create_custom_training_job_from_component(
        component_spec=model_training_task, 
        machine_type=os.environ.get('MACHINE_TYPE'), 
        acclerator_type=os.environ.get('ACCELERATOR_TYPE'),
        accelerator_count=int(os.environ.get('ACCELERATOR_COUNT')),
        replica_count=int(os.environ.get('REPLICA_COUNT')),
        service_account=os.environ.get('SERVICE_ACCOUNT')
    )

    @pipeline(
        name='training-pipeline',
        pipeline_root=os.environ.get('PIPELINE_ROOT')
    )
    def pipeline(
        dataset_name: str, 
        download: bool,
        experiment_name: str,
        epochs: int,
        device: str,
        num_workers: int,
    ):
        """
        Kubeflow pipeline
        """
        
        data_prepare_task_results = (
            data_prepare_task(
                dataset_name=dataset_name,
                download=download
            )
            .set_caching_options(True)
            .set_display_name('Data Prep')
        )

        model_training_task_results = (
            custom_training_job_op(
                experiment_name=experiment_name,
                device=device,
                epochs=epochs,
                trainset=data_prepare_task_results.outputs['trainset'],
                testset=data_prepare_task_results.outputs['testset']
            )
            .set_caching_options(False)
            .set_display_name('Train')
        )

    return pipeline