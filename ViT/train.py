import json
import argparse
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.DataLoader as DataLoader

from dataset import ModelDataset
from model import ViT

config = json.load('config.json')


class Trainer:
    """
    Trains ViT model with PyTorch framework
    """

    def __init__(self, model, optimizer, loss_fn, exp_name, device):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.exp_name = exp_name


    def train(self, epochs, train, test=None):
        """
        Trains the model for a specified number of epochs
        """

        train_losses, test_losses, accuracies = [], [], []

        for i in range(epochs):
            train_loss = self.train_epoch(train)

            accuracy, test_loss = self.evaluate(test)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            print(f'Epochs {i+1} \nTrain Loss: {train_loss} \nTest Loss: {test_loss} \nAccuracy: {accuracy}')


    def train_epoch(self, train: DataLoader):
        """
        Initializes model training for one epoch
        """
        self.model.train()
        total_loss = 0

        for batch in train:
            batch = [t.to(self.device) for t in batch]
            img, label = batch
            # Initialize gradients at zero
            self.optimizer.zero_grad()
            # Calculate loss for the batch
            loss = self.loss_fn(self.model(img)[0], label)
            # Backpropagation
            loss.backward()
            # Update model weights
            self.optimizer.step()
            total_loss += loss.item() * len(img)

        avg_loss = total_loss / len(train.dataset)

        return avg_loss
    

    @torch.no_grad()
    def evaluate(self, test):
        """"
        Evaluates model on test dataset
        """
        self.model.eval()
        total_loss = 0
        correct = 0

        with torch.no_grad():
            for batch in test:
                batch = [t.to(self.device) for t in batch]
                img, label = batch

                # Generate model predictions
                logits, _ =  self.model(img)
                # Calculate loss for the batch
                loss = self.loss_fn(logits, label)
                total_loss += loss.item() * len(img)
                # Obtain accuracy
                preds = torch.argmax(logits, dim=1)
                correct += torch.sum(preds == label).item()

        accuracy = correct / len(test.dataset)
        avg_loss = total_loss / len(test.dataset)

        return accuracy, avg_loss
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()

    dataset = ModelDataset(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    train_transform, test_transform = dataset.transform_data(
        h = 32,
        w = 32,
        p = 0.5
    )
    
    train = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=train_transform
    )

    test = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=test_transform
    )

    trainloader, testloader, _ = dataset.prepare_data(train ,test)

    model = ViT(config=config)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        exp_name=args.exp_name,
        device=args.device
    )

    trainer.train(trainloader, testloader, epochs=args.epochs)



    