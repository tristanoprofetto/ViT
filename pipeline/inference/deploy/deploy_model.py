import argparse
import google.cloud.aiplatform as vertex_ai


def run_deploy_model_task(
    project: str,
    region: str,
    endpoint_id: str,
    model_id: str,
    model_display_name: str,
    machine_type: str,
    accelerator_type: str = "ACCELERATOR_TYPE_UNSPECIFIED",
    acclerator_count: int = 0,
    traffic_split: int = 100,
    min_replica_count: int = 1,
    timeout: int = 7200,
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": f'{region}-aiplatform.googleapis.com'}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = vertex_ai.gapic.EndpointServiceClient(client_options=client_options)
    deployed_model = {
        "model": f'projects/{project}/locations/{region}/models/{model_id}',
        "display_name": model_display_name,
        # `dedicated_resources` must be used for non-AutoML models
        "dedicated_resources": {
            "min_replica_count": min_replica_count,
            "machine_spec": {
                "machine_type": machine_type,
                # Accelerators can be used only if the model specifies a GPU image.
                'accelerator_type': vertex_ai.gapic.AcceleratorType[accelerator_type],
                'accelerator_count': acclerator_count,
            },
        },
    }
    # key '0' assigns traffic for the newly deployed model ... Traffic percentage values must add up to 100
    traffic_split = {"0": traffic_split}
    endpoint = client.endpoint_path(
        project=project, location=region, endpoint=endpoint_id
    )
    response = client.deploy_model(
        endpoint=endpoint, deployed_model=deployed_model, traffic_split=traffic_split
    )
    print("Long running operation:", response.operation.name)
    deploy_model_response = response.result(timeout=timeout)
    print("deploy_model_response:", deploy_model_response)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--project')
    parser.add_argument('--region')
    parser.add_argument('--endpoint_id')
    parser.add_argument('--model_id')
    parser.add_argument('--model_display_name')
    parser.add_argument('--machine_type')
    parser.add_argument('--acclerator_type')
    parser.add_argument('--accelerator_count')
    parser.add_argument('--traffic_split')
    parser.add_argument('--project')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    run_deploy_model_task(
        project=args.project,
        region=args.region,
        endpoint_id=args.endpoint_id,
        model_id=args.model_id,
        model_display_name=args.model_display_name,
        machine_type=args.machine_type,
        accelerator_type=args.accelerator_type,
        acclerator_count=args.accelerator_count,
        traffic_split=args.traffic_split
    )