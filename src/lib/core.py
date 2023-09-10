import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import src.lib.metrics as metrics

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test_model_online(model: nn.Module, test_loader: torch.utils.data.DataLoader, loss_function, online = False):
    """
    Loads a neural net to make predictions

    Parameters:
    ===========
    model: model that we want to test
    test_loader: pytorch data loader wrapping test data
    loss_function: loss funct used to evaluate the model
    """
    metric = metrics.calculate_mean_loss_function_online
    test_loss = metric(model, test_loader, loss_function, 1.0)
    print(f"Test Loss: {test_loss}")

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
