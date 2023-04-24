import argparse
from typing import Dict, Optional, Sequence

import google.cloud.aiplatform as vertex_ai


def run_upload_model_task(
    project: str,
    location: str,
    description: str,
    model_display_name: str,
    model_artifact_uri: str,
    serving_container_image_uri: str,
    serving_container_predict_route: str,
    serving_container_health_route: str,
    serving_container_command: Optional[Sequence[str]] = None,
    serving_container_args: Optional[Sequence[str]] = None,
    serving_container_environment_variables: Optional[Dict[str, str]] = None,
    serving_container_ports: Optional[Sequence[int]] = None,
):
    """
    Uploads model to Vertex AI Model Registry
    """

    model = vertex_ai.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_artifact_uri,
        serving_container_image_uri=serving_container_image_uri,
        serving_container_predict_route=serving_container_predict_route,
        serving_container_health_route=serving_container_health_route,
        description=description,
        serving_container_command=serving_container_command,
        serving_container_args=serving_container_args,
        serving_container_environment_variables=serving_container_environment_variables,
        serving_container_ports=serving_container_ports
    )

    model.wait()

    return model


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--region')
    parser.add_argument('--model_artifact_uri')
    parser.add_argument('--serving_image_uri')
    parser.add_argument('--health_route')
    parser.add_argument('--predict_route')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()