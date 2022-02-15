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


def train_model_offline(
    net: nn.Module,
    path: str,
    parameters: dict,
    train_loader: DataLoader,
    validation_loader: DataLoader = None,
    name: str = "Model",
    logger: TrainLogger = None,
    snapshot_iterations: int = None
) -> dict:
    """
    Trains and saves a neural net

    Parameters:
    ===========
    net: Module representing a neural net to train
    path: dir where models are going to be saved
    parameters: dict having the following data:
                - "lr": learning rate
                - "momentum": momentum of the optimizer
                - "criterion": loss function
                - "epochs": epochs to train
    train_loader: pytorch DataLoader wrapping training set
                  This MUST be in the form of triplets: (anchor, positive, negative)
    validation_loader: pytorch DataLoader wrapping validation set
                       This MUST be in the form of triplets: (anchor, positive, negative)
    name: name of the model, in order to save it
    train_logger: to log data about trainning process
                  Default logger is silent logger
    snapshot_iterations: at how many iterations we want to take an snapshot of the model
                         If its None, no snapshots are taken
    """

    # Loss and optimizer
    lr = parameters["lr"]
    criterion = parameters["criterion"]

    # Use Adam as optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr)

    # Select proper device and move the net to that device
    device = get_device()
    net.to(device)

    # Check if no logger is given
    if logger is None:
        print("==> No logger given, using Silent Logger")
        logger = SilentLogger()

    # Printing where we're training
    print(f"==> Training on device {device}")
    print("")

    # Dict where we are going to save the training history
    training_history = dict()
    training_history["loss"] = []
    training_history["val_loss"] = []

    # Training the network
    epochs = parameters["epochs"]
    for epoch in range(epochs):

        for i, data in enumerate(train_loader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = [net(item[None, ...].to(device)) for item in data]
            loss = criterion(outputs)

            # Backward + Optimize
            loss.backward()
            optimizer.step()

            # Statistics -- important, we have to use the iteration given by current epoch and current
            # iteration counter in inner loop. Otherwise logs are going to be non-uniform over iterations
            curr_it = epoch * len(train_loader.dataset) + i * train_loader.batch_size
            if logger.should_log(curr_it):
                # Log and return loss from training and validation
                training_loss, validation_loss = logger.log_process(train_loader, validation_loader, epoch, i)

                # Save loss of training and validation sets
                training_history["loss"].append(training_loss)
                training_history["val_loss"].append(validation_loss)

            # Snapshots -- same as Statistics, we reuse current iteration calc
            # TODO -- create a SnapshotTaker class as we have for logs -- snapshot_taker.should_log(i)
            if snapshot_iterations is not None and curr_it % snapshot_iterations == 0:
                # We take the snapshot
                snapshot_name = "snapshot_" + name + "==" + get_datetime_str()
                snapshot_folder = os.path.join(path, "snapshots")
                filesystem.save_model(net, folder_path = snapshot_folder, file_name = snapshot_name)

    print("Finished training")


    # Save the model -- use name + date stamp to save the model
    date = get_datetime_str()
    name = name + "==" + date
    filesystem.save_model(net = net, folder_path = path, file_name = name)

    # Return the training hist
    return training_history


def train_model_online(
    net: nn.Module,
    path: str,
    parameters: dict,
    train_loader: DataLoader,
    validation_loader: DataLoader = None,
    name: str = "Model",
    logger: TrainLogger = None,
    snapshot_iterations: int = None
) -> dict:
    """
    Trains and saves a neural net

    Parameters:
    ===========
    net: Module representing a neural net to train
    path: dir where models are going to be saved
    parameters: dict having the following data:
                - "lr": learning rate
                - "momentum": momentum of the optimizer
                - "criterion": loss function
                - "epochs": epochs to train
    train_loader: pytorch DataLoader wrapping training set
                  This MUST NOT be in the form of triplets: (anchor, positive, negative)
    validation_loader: pytorch DataLoader wrapping validation set
                       This MUST NOT be in the form of triplets: (anchor, positive, negative)
    name: name of the model, in order to save it
    train_logger: to log data about trainning process
                  Default logger is silent logger
    snapshot_iterations: at how many iterations we want to take an snapshot of the model
                         If its None, no snapshots are taken
    """

    # Loss and optimizer
    lr = parameters["lr"]
    criterion = parameters["criterion"]

    # Use Adam as optimizer
    optimizer = optim.Adam(net.parameters(), lr = lr)

    # Select proper device and move the net to that device
    device = get_device()
    net.to(device)

    # Check if no logger is given
    if logger is None:
        print("==> No logger given, using Silent Logger")
        logger = SilentLogger()

    # Printing where we're training
    print(f"==> Training on device {device}")
    print("")

    # Dict where we are going to save the training history
    training_history = dict()
    training_history["loss"] = []
    training_history["val_loss"] = []

    # For controlling the logging
    current_seen_batches = 0

    # Training the network
    epochs = parameters["epochs"]
    for epoch in range(epochs):

        for i, data in enumerate(train_loader):

            # Unwrap the data
            imgs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = net(imgs.to(device))
            loss = criterion(outputs, labels.to(device))

            # Backward + Optimize
            loss.backward()
            optimizer.step()

            # Update counter
            current_seen_batches += 1
            #  print(f"TODO -- current seen batches = {current_seen_batches}")

            # Statistics -- important, we have to use the iteration given by current epoch and current
            # iteration counter in inner loop. Otherwise logs are going to be non-uniform over iterations
            curr_it = epoch * len(train_loader.dataset) + i * train_loader.batch_size

            #  if logger.should_log(curr_it) or current_seen_batches == 10:
            if current_seen_batches == 10:
                # Log and return loss from training and validation
                training_loss, validation_loss = logger.log_process(train_loader, validation_loader, epoch, i)

                # Save loss of training and validation sets
                training_history["loss"].append(training_loss)
                training_history["val_loss"].append(validation_loss)
            else:
                print(f"[{epoch} / {curr_it}]")

            # Snapshots -- same as Statistics, we reuse current iteration calc
            # TODO -- create a SnapshotTaker class as we have for logs -- snapshot_taker.should_log(i)
            if snapshot_iterations is not None and curr_it % snapshot_iterations == 0:
                # We take the snapshot
                snapshot_name = "snapshot_" + name + "==" + get_datetime_str()
                snapshot_folder = os.path.join(path, "snapshots")
                filesystem.save_model(net, folder_path = snapshot_folder, file_name = snapshot_name)

            # Re-start counter if hit 10
            current_seen_batches = current_seen_batches % 10


    print("Finished training")


    # Save the model -- use name + date stamp to save the model
    date = get_datetime_str()
    name = name + "==" + date
    filesystem.save_model(net = net, folder_path = path, file_name = name)

    # Return the training hist
    return training_history
