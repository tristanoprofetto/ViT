from kfp.v2.dsl import component, Input, Output, Model, Metrics
from training.train import Trainer
from training.model import ViT


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

    with open(trainset.path, 'r') as file:
       train_data = file.read()

    with open(testset.path, 'r') as file:
       test_data = file.read()

    trainer = Trainer(
        model = ViT,
        optimizer=optimzer,
        loss_fn=loss_fn,
        device=device
    )

    trainer.train(
       train=train_data, 
       test=test_data, 
       epochs=epochs
    )

    accuracy, loss = trainer.evaluate(test=test_data)
    
    metrics.log_metric('accuracy', accuracy)
    metrics.log_metric('loss', loss)

    

    
