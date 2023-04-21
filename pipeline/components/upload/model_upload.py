from typing import Dict, Optional, Sequence

import google.cloud.aiplatform as aip


def upload_model(
    project: str,
    location: str,
    model_display_name: str,
    artifact_uri: Optional[str],
    serving_container_image_uri: str,
    serving_container_predict_route: Optional[str] = None,
    serving_container_health_route: Optional[str] = None,
    description: Optional[str] = None,
    serving_container_command: Optional[Sequence[str]] = None,
    serving_container_args: Optional[Sequence[str]] = None,
    serving_container_environment_variables: Optional[Dict[str, str]] = None,
    serving_container_ports: Optional[Sequence[int]] = None,
):
    """
    Uploads model to Vertex AI Model Registry
    """

    aip.init(
        project=project, 
        location=location
    )

    model = aip.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
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