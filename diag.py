import os
import argparse

from diagrams import Diagram, Cluster
from diagrams.gcp.ml import AIPlatform
from diagrams.gcp.devtools import ContainerRegistry
from diagrams.onprem.workflow import Kubeflow
from diagrams.onprem.container import Docker
from diagrams.onprem.vcs import Git
from diagrams.programming.language import Bash, Python


def create_diagram(title: str, filepath: str):

    with Diagram(title):

        with Cluster("Training Piplines"):
            data_prep = AIPlatform("Data Prep")
            train = AIPlatform("Model Training")

        with Cluster("Inference Pipeline"):
            upload = AIPlatform("Model Upload")
            deploy = AIPlatform("Model Deploy")

        data_prep >> train >> upload >> deploy


def create_training_diagram(title: str, filepath: str):

    with Diagram(title):
        
        with Cluster("Automated CI/CD Pipeline"):
            kickoff = Git("On Push")
            script = Bash("Execute Training script")
            docker_image = Docker("Build Training Image")
            artifact_image = ContainerRegistry("Push training image")
            execute_pipeline = AIPlatform("Execute training pipeline")


        with Cluster("Training Piplines"):
            data_prep = Kubeflow("Data Prep")
            train = Kubeflow("Model Training")

        kickoff >> script
        script >> docker_image >> artifact_image
        script >> execute_pipeline
        execute_pipeline >> data_prep >> train 


if __name__ == "__main__":
    create_diagram()





