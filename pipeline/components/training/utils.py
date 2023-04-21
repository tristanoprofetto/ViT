import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms

from training.model import ViT


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
    def visualize_attention(self, model, num_samples, output=None, device='cpu'):
        """
        Visualize attention maps for specified number of images
        """
        model.eval()
        # Load random images
        num_images = 30
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        # Pick 30 samples randomly
        indices = torch.randperm(len(testset))[:num_images]
        raw_images = [np.asarray(testset[i][0]) for i in indices]
        labels = [testset[i][1] for i in indices]
        # Convert the images to tensors
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        images = torch.stack([test_transform(image) for image in raw_images])
        # Move the images to the device
        images = images.to(device)
        model = model.to(device)
        # Get the attention maps from the last block
        logits, attention_maps = model(images, output_attentions=True)
        # Get the predictions
        predictions = torch.argmax(logits, dim=1)
        # Concatenate the attention maps from all blocks
        attention_maps = torch.cat(attention_maps, dim=1)
        # select only the attention maps of the CLS token
        attention_maps = attention_maps[:, :, 0, 1:]
        # Then average the attention maps of the CLS token over all the heads
        attention_maps = attention_maps.mean(dim=1)
        # Reshape the attention maps to a square
        num_patches = attention_maps.size(-1)
        size = int(math.sqrt(num_patches))
        attention_maps = attention_maps.view(-1, size, size)
        # Resize the map to the size of the image
        attention_maps = attention_maps.unsqueeze(1)
        attention_maps = F.interpolate(attention_maps, size=(32, 32), mode='bilinear', align_corners=False)
        attention_maps = attention_maps.squeeze(1)
        # Plot the images and the attention maps
        fig = plt.figure(figsize=(20, 10))
        mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
        for i in range(num_images):
            ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
            img = np.concatenate((raw_images[i], raw_images[i]), axis=1)
            ax.imshow(img)
            # Mask out the attention map of the left image
            extended_attention_map = np.concatenate((np.zeros((32, 32)), attention_maps[i].cpu()), axis=1)
            extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
            ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
            # Show the ground truth and the prediction
            gt = classes[labels[i]]
            pred = classes[predictions[i]]
            ax.set_title(f"gt: {gt} / pred: {pred}", color=("green" if gt==pred else "red"))
        if output is not None:
            plt.savefig(output)
        plt.show()



