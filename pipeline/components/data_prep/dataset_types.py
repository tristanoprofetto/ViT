import os
from torchvision import datasets


def retrieve_dataset(
        name, 
        download, 
        train_transform, 
        test_transform
):
    """
    Gets torchvision dataset by name

    Args:
        name
        download
        train_transform
        test_transform
    """
    
    if name == "CIFAR10":
        train = datasets.CIFAR10(
            root='./',
            train=True,
            transform=train_transform,
            download=download
        )

        test = datasets.CIFAR10(
            root='./',
            train=False,
            transform=test_transform,
            download=download
        )

    elif name == "CIFAR100":
        train = datasets.CIFAR100(
            root='./',
            train=True,
            transform=train_transform,
            download=download
        )

        test = datasets.CIFAR100(
            root='./',
            train=False,
            transform=test_transform,
            download=download
        )

    elif name == "Food101":
        train = datasets.Food101(
            root='./',
            train=True,
            transform=train_transform,
            download=download
        )

        test = datasets.Food101(
            root='./',
            train=False,
            transform=test_transform,
            download=download
        )

    elif name == "Flower102":
        train = datasets.Flower102(
            root='./',
            train=True,
            transform=train_transform,
            download=download
        )

        test = datasets.Flower102(
            root='./',
            train=False,
            transform=test_transform,
            download=download
        )

    elif name == "ImageNet":
        train = datasets.ImageNet(
            root='./',
            train=True,
            transform=train_transform,
            download=download
        )

        test = datasets.ImageNet(
            root='./',
            train=False,
            transform=test_transform,
            download=download
        )

    elif name == "MNIST":
        train = datasets.MNIST(
            root='./',
            train=True,
            transform=train_transform,
            download=download
        )

        test = datasets.MNIST(
            root='./',
            train=False,
            transform=test_transform,
            download=download
        )

    elif name == "SUN397":
        train = datasets.SUN397(
            root='./',
            train=True,
            transform=train_transform,
            download=download
        )

        test = datasets.SUN397(
            root='./',
            train=False,
            transform=test_transform,
            download=download
        )
    
    return train, test
