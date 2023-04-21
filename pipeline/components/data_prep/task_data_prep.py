import os
import json
import argparse
import configparser
import torch
from torchvision import transforms, datasets

from data_prep.dataset_types import retrieve_dataset

config = configparser.ConfigParser()
config.read('transforms.ini')


def run_data_prep_task(
    dataset_name: str,
    train_path: str, 
    test_path: str, 
    download: bool, 
):
    """
    Data preparation: transforms images into tensors

    Args:
        dataset_name (str): name of torchvision dataset
        train_path (str):
        test_path (str):
        download (bool):
    """
    mean = config.getfloat('Normalize', 'mean')
    std = config.getfloat('Normalize', 'std')

    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((config.getint('Resize', 'h'), config.getint('Resize', 'w'))),
            transforms.RandomHorizontalFlip(p=config.getfloat('RandomHorizontalFlip', 'p')),
            transforms.RandomResizedCrop(
                size=(config.getint('RandomResizedCrop', 'h'), config.getint('RandomResizedCrop', 'w')), 
                scale=(config.getfloat('RandomResizedCrop', 'scale_min'), config.getfloat('RandomResizedCrop', 'scale_max')), 
                ratio=(config.getfloat('RandomResizedCrop', 'ratio_min'), config.getfloat('RandomResizedCrop', 'ratio_max')), 
                interpolation=config.getint('RandomResizedCrop', 'interpolation')),
            transforms.Normalize((mean, mean, mean), (std, std, std))
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((config.getint('Resize', 'h'), config.getint('Resize', 'w'))),
            transforms.Normalize((mean, mean, mean), (std, std, std))
        ]
    )

    # Download/retrieve torchvision datasets
    train, test = retrieve_dataset(
        name=dataset_name, 
        download=download, 
        train_transform=train_transform, 
        test_transform=test_transform
    )

    # Save datasets to Ouput artifact destination paths
    torch.save(train, f'{train_path}/train.pt')
    torch.save(test, f'{test_path}/test.pt')


def get_args():
    """
    Get command-line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--download', type=bool, default=True)

    args = parser.parse_args()
    
    return args


if __name__ == "__main__":

    args = get_args()

    run_data_prep_task(
        dataset_name=args.dataset_name,
        train_path=args.train_path,
        test_path=args.test_path,
        download=args.download
    )
