import os
from typing import Optional
from configparser import ConfigParser

from kfp.dsl import components, pipeline
from kfp.v2.dsl import component
from kp.v2.components import importer_node
from google_cloud_pipeline_components.types import artifact_types

model_upload_task = components.load_component_from_file()
model_deploy_task = components.load_component_from_file()


def build_inference_pipeline(cfg: ConfigParser):
    """
    Builds the Inference Pipeline
    """
    model_upload_op = components.load_component_from_file('./component_configs/model_upload_component.yaml')
    endpoint_create_op = components.load_component_from_file('./component_configs/endpoint_create_component.yaml')
    model_deploy_op = components.load_component_from_file('./component_configs/model_deploy_component.yaml')

    @pipeline(
        name=cfg.get('pipeline_inference', 'name'),
        pipeline_root=os.environ.get('PIPELINE_ROOT')
    )
    def pipeline(
        model_display_name: str, 
        model_artifact_uri: str,
        endpoint_artifact_uri: str,
        serving_image_uri: str,
        health_route: str = cfg.get('model_upload', 'predict_route'),
        predict_route: str = cfg.get('model_upload', 'health_route'),
        machine_type: str = cfg.get('model_deploy', 'machine_type'),
        accelerator_type: str = cfg.get('model_deploy', 'accelerator_type'),
        accelerator_count : int = cfg.getint('model_deploy', 'accelerator_count'),
        traffic_percentage: int = cfg.getint('model_deploy', 'traffic_percentage'),
        autoscaling_nodes_min: int = cfg.getint('model_deploy', 'min_replica_count'),
        autoscaling_nodes_max: int = cfg.getint('model_deploy', 'max_replica_count'),
        autoscaling_cpu_utilization: int = cfg.getint('model_deploy', 'cpu_utilization'),
        serving_container_command: Optional[list] = None,
        serving_container_environment_variables: Optional[dict] = None,
    ):
        """
        Kubeflow pipeline
        """
        
        model_upload_op_results = (
            model_upload_op(
                model_display_name=model_display_name, 
                model_artifact_uri=model_artifact_uri,
                serving_container_image_uri=serving_image_uri,
                serving_container_predict_route=predict_route,
                serving_container_health_route=health_route,
                serving_container_command=serving_container_command,
                serving_container_environment_variables=serving_container_environment_variables,
            )
        )

        import_endpoint_task = (
            importer_node.importer(
                artifact_uri=endpoint_artifact_uri,
                artifact_class=artifact_types.VertexEndpoint,
                metadata={
                    "resourceName": os.environ.get('DEPLOY_MODEL_ENDPOINT')
                }
            )
                .set_caching_options(False)
                .set_display_name('Endpoint Artifact Import')

        )

        model_deploy_op_result = (
            model_deploy_op(
                endpoint=import_endpoint_task.outputs['artifact'],
                model=model_upload_op_results.outputs['model'],
                model_display_name=model_display_name,
                machine_type=machine_type,
                accelerator_type=accelerator_type,
                accelerator_count=accelerator_count,
                traffic_percentage=traffic_percentage,
                autoscaling_nodes_min=autoscaling_nodes_min,
                autoscaling_nodes_max=autoscaling_nodes_max,
                autoscaling_cpu_utilization=autoscaling_cpu_utilization

            )
        )


    return pipeline
