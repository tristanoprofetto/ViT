import argparse

import google.cloud.aiplatform as vertex_ai


def run_endpoint_create_task(
        project: str, 
        region: str,
        endpoint_display_name: str,
        description: str,
):
    """
    Creates an Enpoint Artifact in Vertex AI
    """

    endpoint = vertex_ai.Endpoint.create(
        project=project,
        location=region,
        display_name=endpoint_display_name,
        description=description
    )

    print(f'Created Vertex AI Endpoint : {endpoint_display_name}')


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--region')
    parser.add_argument('--endpoint_display_name')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()