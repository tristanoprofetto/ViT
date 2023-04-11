import torch
import torchvision
import torch.nn as nn
import torcchvision.transforms as transforms
import torch.utils.data.Dataset as Dataset
import torch.utils.data.DataLoader as DataLoader


class ModelDataset(nn.Module):
    """
    Prepares dataset suitable to be input into the Encoder block
    """

    def __init__(self, batch_size: int, num_workers: int, train_size: int = None, test_size: int = None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_size = train_size
        self.test_size = test_size


    def transform_data(self, h: int, w: int, p: float):
        """"
        Defines the necessary image transformations

        Args:
            h (int): height of desired output images
            w (int): width of desired output images
            p (float): probability of image being flipped

        Returns:
            train_transform (torchvision.transforms.Compose(list)): image transformations for training set
            test_transform (torchvision.transforms.Compose(list)): image transformations for test set
        """

        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((h, w)),
                transforms.RandomHorizontalFlip(p=p),
                transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((h, w)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        return train_transform, test_transform
    

    def load_dataset(self, test: bool=False):
        """
        Loads PyTorch dataset
        """
        train_transform, test_transform = self.transform_data(h=32, w=32, p=0.5)
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

        return train, test


    def prepare_data(self, train: Dataset, test: Dataset=None):
        """
        Loads torch Dataset into torch Dataloader object
        """
        train_loader = DataLoader(
            train, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

        if test is not None:
            test_loader = DataLoader(
                test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )

            return train_loader, test_loader
        
        else:
            return train_loader
        

    def sample_dataset(self, input_dataset: Dataset, sample_size: int):
        """
        Samples the dataset on a subset of the input data
        """

        indices = torch.randperm(len(input_dataset))[:sample_size]
        sampled_dataset = torch.utils.data.Subset(input_dataset, indices)

        return sampled_dataset



















