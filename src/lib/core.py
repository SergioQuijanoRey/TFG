import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

import board
import filesystem
from train_loggers import TrainLogger, SilentLogger
import metrics

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# TODO -- use metrics module to calculate this metrics
def test_model(model: nn.Module, test_loader: torch.utils.data.DataLoader, loss_function, online = False):
    """
    Loads a neural net to make predictions

    Parameters:
    ===========
    model: model that we want to test
    test_loader: pytorch data loader wrapping test data
    loss_function: loss funct used to evaluate the model
    """
    metric = metrics.calculate_mean_triplet_loss_offline

    test_loss = metric(model, test_loader, loss_function)
    print(f"Test Loss: {test_loss}")

def test_model_online(model: nn.Module, test_loader: torch.utils.data.DataLoader, loss_function, online = False):
    """
    Loads a neural net to make predictions

    Parameters:
    ===========
    model: model that we want to test
    test_loader: pytorch data loader wrapping test data
    loss_function: loss funct used to evaluate the model
    """
    metric = metrics.calculate_mean_triplet_loss_online
    test_loss = metric(model, test_loader, loss_function, 1.0)
    print(f"Test Loss: {test_loss}")


def split_train_test(dataset, train_percentage: float = 0.8):
    """
    Splits a pytorch dataset into train / test datasets

    Parameters:
    ===========
    dataset: the pytorch dataset
    train_percentage: percentage of the dataset given to train

    Returns:
    ========
    train_dataset
    test_dataset
    """

    # Calculate sizes
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_percentage)
    test_size = dataset_size - train_size

    # Split and return
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_dataset, test_dataset

def get_device() -> str:
    """
    Returns the most optimal device available
    ie. if gpu available, return gpu. Else return cpu

    Returns:
    ========
    device: string representing the device
    """

    if torch.cuda.is_available():
      device = "cuda:0"
    else:
      device = "cpu"
    return device

def get_datetime_str() -> str:
    """Returns a string having a date/time stamp"""
    return datetime.now().strftime("%d-%m-%Y--%H:%M:%S")
