import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import ViT


class ModelOps:
    """
    Allows several functions during and after model training
    """

    def __init__(self, experiment_name):
        super().__init__()
        self.experiment_name = experiment_name


    def save_checkpoint(self, base_dir, model, epoch):
        """
        Saves the model parameters at a specified epoch
        """
        output_dir = os.path.join(base_dir, self.experiment_name)
        os.makedirs(output_dir, exist_ok=True)
        copy = os.path.join(output_dir, f'model_epoch_{epoch}.pt')
        torch.save(model.state_dict(), copy)


    def save_experiment(self, base_dir, train_losses, test_losses, accuracies):
        """
        Saves necessary training artifacts to a specified output directory
        """
        output_dir = os.path.join(base_dir, self.experiment_name)
        os.makedirs(output_dir, exist_ok=True)

        config_file = os.path.join(output_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, sort_keys=True)

        metrics_file = os.path.join(output_dir, 'metric.json')
        with open(metrics, 'w') as f:
            metrics = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'accuracies': accuracies
            }
            json.dump(metrics, f, sort_keys=True)
        
        self.save_checkpoint(self.experiment_name, model)


    def load_experiment(self, base_dir, checkpoint):
        """
        Loads training artifacts from checkpoint directory
        """

        artifacts_dir = os.path.join(base_dir, self.experiment_name)

        config_file = os.path.join(artifacts_dir, 'config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)

        metrics_file = os.path.join(artifacts_dir, 'metrics.json')
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        model = ViT(config)
        model_params = os.path.join(artifacts_dir, checkpoint)
        model.load_state_dict(torch.load(model_params))
        
        return model, config, metrics


    def visualize_data(self, dataset, classes, num_samples):
        """
        Visualize input data
        """

        indices = torch.randperm(len(dataset))[:num_samples]
        images = [np.asarray(dataset[i][0]) for i in indices]
        labels = [dataset[i][1] for i in indices]

        fig = plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
            ax.imshow(images[i])
            ax.set_title(classes[labels[i]])
        

    @torch.no_grad()
    def visualize_attention(self, model, num_samples):
        """
        Visualize attention maps for specified number of images
        """
        pass



