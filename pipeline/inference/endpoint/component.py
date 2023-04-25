from kfp.v2.dsl import component, Input, Output, Model, Metrics, Dataset, Artifact
from pipeline.inference.endpoint.endpoint_create import run_endpoint_create_task

@component(
    base_image='',
    output_component_file='component_configs/endpoint_create_component.yaml'
)
def data_preperation_component(
    endpoint_display_name: str,
    endpoint: Output[Artifact]
):
    """
    Endpoint creation component

    Args:
        endpoint_display_name (str): 
        model_artifact_uri (str):
        serving_image_uri (str):
        serving_image_predict_route (str):
        endpoint (Output[Artifact]):

    Returns:

    """

    try:
        endpoint = run_endpoint_create_task(
            endpoint_display_name=endpoint_display_name
        )

        with open(endpoint.path, 'w') as writer:
            writer.write(endpoint)

    except Exception as e:
        print(e)




    