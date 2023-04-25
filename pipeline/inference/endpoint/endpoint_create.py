import argparse
from typing import Optional
import google.cloud.aiplatform as vertex_ai


def run_endpoint_create_task(
        endpoint_display_name: str,
        description: Optional[str],
):
    """
    Creates an Endpoint Artifact in Vertex AI

    Args:
        endpoint_display_name (str)

    Returns:
        endpoint (google.cloud.aiplatform.Endpoint):
    """

    endpoint = vertex_ai.Endpoint.create(
        display_name=endpoint_display_name,
    )

    print(f'Created Vertex AI Endpoint : {endpoint_display_name}')
    return endpoint


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint_display_name')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    run_endpoint_create_task(
        endpoint_display_name=args.endpoint_display_name
    )