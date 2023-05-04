import os
import json
import argparse

import torch
from torch import nn, optim

from pipeline.training.train_model.dataset import ModelDataset
from pipeline.training.train_model.train import Trainer
from pipeline.training.train_model.model import ViT


def run_train_model_task(
    model_name: str,
    train_path: str,
    test_path: str,
    experiment_name: str,
    device: str,
    epochs: int,
    num_workers: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    model_artifacts_path: str,
    metrics_artifacts_path: str,
):
    """
    Model training

    Args:
        model_name (str): name of torchvision dataset
        train_path (str): path to the input dataset for model training
        test_path (str): path to the input dataset for model evaluation
        experiment_name (str):
        device (str): hardware for model training cpu/gpu
        epochs (int): number of training iterations
        workers (int)
        batch_size (int)
        learning_rate (float)
        weight_decay (float)
        model_artifacts_path (str): file path to save trained model artifacts
        metrics_artifacts_path (str): file path to save test metrics
    """

    config = json.load('./pipeline/training/train_model/config.json')

    with open(train_path, 'r') as file:
       train = file.read()

    with open(test_path, 'r') as file:
       test = file.read()

    dataset = ModelDataset(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    trainloader, testloader, _ = dataset.prepare_data(train ,test)

    model = ViT(config=config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    trainer = Trainer(
        model=model,
        experiment_name=experiment_name,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device
    )

    trainer.train(
        train=train, 
        test=test, 
        epochs=epochs
    )

    trainer.train(trainloader, testloader, epochs=epochs)

    accuracy, loss = trainer.evaluate(test=test)

    torch.save(model.state_dict(), f'{model_artifacts_path}/{model_name}')

    with open(metrics_artifacts_path, 'w') as w:
        w.write({'accuracy': accuracy, 'loss': loss})


def get_args():
    """
    Get command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--epochs', type=int, default=True)

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":
    args = get_args()
