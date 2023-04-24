import argparse

import google.cloud.aiplatform as vertex_ai


def deploy_model_task(
        endpoint: vertex_ai.Endpoint,
        model: vertex_ai.models.Model,
        model_display_name: str,
        machine_type: str,
        accelerator_type: str,
        accelerator_count : int,
        traffic_split: int,
        autoscaling_nodes_min: int,
        autoscaling_nodes_max: int,
        autoscaling_cpu_utilization: int
):
    """
    Deploys Model to Endpoint
    """

    endpoint.deploy(
        model=model,
        deployed_model_display_name=model_display_name,
        machine_type=machine_type,
        accelerator_type=accelerator_type,
        accelerator_count=accelerator_count,
        traffic_percentage=traffic_split,
        min_replica_count=autoscaling_nodes_min,
        max_replica_count=autoscaling_nodes_max,
        autoscaling_target_cpu_utilization=autoscaling_cpu_utilization

    )


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint')
