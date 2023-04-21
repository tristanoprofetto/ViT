import os
from kfp.dsl import components, pipeline
from kfp.v2.dsl import component

model_upload_task = components.load_component_from_file()
model_deploy_task = components.load_component_from_file()

@pipeline(
    name='deploy-model',
    pipeline_root=os.environ.get('PIPELINE_ROOT')
)
def deployment_pipeline(
    model_display_name: str, 
    model_artifact_uri: str,
    serving_image_uri: str,
    health_route: str,
    predict_route: str,
):
    """
    Kubeflow pipeline
    """
    pass