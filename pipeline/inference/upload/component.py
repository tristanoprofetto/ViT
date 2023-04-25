from kfp.v2.dsl import component, Input, Output, Model, Metrics, Dataset
from pipeline.inference.upload.model_upload import run_model_upload_task

@component(
    base_image='',
    output_component_file='component_configs/upload_model_component.yaml'
)
def data_preperation_component(
    model_display_name: str,
    model_artifact_uri: str,
    serving_image_uri: str,
    serving_container_predict_route: str,
    serving_container_health_route: str,
    serving_container_ports: list,
    model: Output[Model]
):
    """
    Model Upload component

    Args:
        model_display_name (str): 
        model_artifact_uri (str):
        serving_image_uri (str):
        serving_image_predict_route (str):
        model (Output[Model]):

    Returns:

    """

    try:
        model = run_model_upload_task(
            model_artifact_uri=model_artifact_uri.path,
            serving_image_uri=serving_image_uri,
            serving_container_predict_route=serving_container_predict_route,
            serving_container_health_route=serving_container_health_route,
            serving_container_ports=serving_container_ports
            
        )

        with open(model.path, 'w') as writer:
            writer.write(model)

    except Exception as e:
        print(e)




    