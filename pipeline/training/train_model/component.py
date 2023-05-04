from kfp.v2.dsl import component, Input, Output, Model, Metrics

from pipeline.training.train_model.task_train_model import run_train_model_task


@component(
      base_image='',
      output_component_file='component_configs/model_training_component.yaml'
)
def model_training_component(
    model_name: str,
    experiment_name: str,
    device: str,
    epochs: int,
    num_workers: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    trainset: Input[Dataset],
    testset: Input[Dataset],
    metrics: Output[Metrics],
    model: Output[Model]
):
    """
    Model Training component

    Args:
        model_name (str): 
        experiment_name (str):
        device (str): CPU/CPU
        epochs: number of training iterations
        num_workers (int):
        batch_size (int):
        learning_rate (float):
        weight_decay (float):


    Returns:

    """
    try:
        run_train_model_task(

        )
    except Exception as e:
        print(e)

    

    
