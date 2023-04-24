from kfp.v2.dsl import component, Input, Output, Model, Metrics, Dataset
from pipeline.training.data_prep.task_data_prep import run_data_prep_task

@component(
    base_image='',
    output_component_file='component_configs/data_preperation_component.yaml'
)
def data_preperation_component(
    dataset_name: str,
    download: bool,
    trainset: Output[Dataset],
    testset: Output[Dataset]
):
    """
    Model Training component

    Args:
        dataset_name (str): 
        download (bool):
        trainset (Output[Dataset]):
        testset (Output[Dataset]):

    Returns:

    """

    try:
        run_data_prep_task(
            dataset_name=dataset_name,
            download=download,
            train_path=trainset.path,
            test_path=testset.path
        )

    except Exception as e:
        print(e)




    