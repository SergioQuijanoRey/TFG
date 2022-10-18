"""
Code for different types of training
"""


import logging
import os
from typing import List, Dict

import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

import src.lib.filesystem as filesystem
from src.lib.core import get_device, get_datetime_str
from src.lib.train_loggers import TrainLogger, SilentLogger

file_logger = logging.getLogger("MAIN_LOGGER")

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

    # Get parameters from the dic parameter
    lr = parameters["lr"]
    criterion = parameters["criterion"]
    epochs = parameters["epochs"]

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
    # Loggers need to know how many single elements we have seen
    # We have to compute it adding up every time a new batch is seen. Because of P-K sampling
    # different epochs mean different number of elements seen
    how_may_elements_seen = 0

    # Training the network
    for epoch in range(epochs):

        # Logger needs to know the parameter `epoch_iteration`
        # This, as `TrainLogger(ABC)` documentation says, it's the number of seen single elements
        # in this epoch. Thus, it's not the same as `how_may_elements_seen`.
        # And for the same reason as `how_may_elements_seen`, we have to compute it summing up
        # manually
        epoch_iteration = 0

        for i, data in enumerate(train_loader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = [net(item[None, ...].to(device)) for item in data]
            loss = criterion(outputs)

            # Backward + Optimize
            loss.backward()
            optimizer.step()

            # Update the number counter for seen elements
            how_may_elements_seen += len(outputs)
            epoch_iteration += len(outputs)

            if logger.should_log(how_may_elements_seen):

                # Log and return loss from training and validation
                training_loss, validation_loss = logger.log_process(
                    train_loader,
                    validation_loader,
                    epoch,
                    epoch_iteration
                )

                # Save loss of training and validation sets
                training_history["loss"].append(training_loss)
                training_history["val_loss"].append(validation_loss)
            else:
                print(f"[{epoch} / {epoch_iteration}]")

            # Snapshots -- same as Statistics, we reuse current iteration calc
            # TODO -- create a SnapshotTaker class as we have for logs -- snapshot_taker.should_log(i)
            if snapshot_iterations is not None and how_may_elements_seen % snapshot_iterations == 0:

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
) -> Dict[str, List[float]]:
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

    # Get parameters from the dic parameter
    lr = parameters["lr"]
    criterion = parameters["criterion"]
    epochs = parameters["epochs"]

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

    # Dict where we are going to save the values of the metrics returned by the logger
    training_history = dict()

    # For controlling the logging
    # Loggers need to know how many single elements we have seen
    # We have to compute it adding up every time a new batch is seen. Because of P-K sampling
    # different epochs mean different number of elements seen
    how_may_elements_seen = 0

    # Training the network
    for epoch in range(epochs):

        # Logger needs to know the parameter `epoch_iteration`
        # This, as `TrainLogger(ABC)` documentation says, it's the number of seen single elements
        # in this epoch. Thus, it's not the same as `how_may_elements_seen`.
        # And for the same reason as `how_may_elements_seen`, we have to compute it summing up
        # manually
        epoch_iteration = 0

        file_logger.info(f"Started epoch {epoch}")

        for i, data in enumerate(train_loader, 0):

            # Unwrap the data
            imgs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = net(imgs.to(device))
            loss = criterion(outputs, labels.to(device))
            file_logger.debug(f"Obtained running loss at {i} iteration of epoch {epoch} is {loss}")

            # Backward + Optimize
            loss.backward()
            optimizer.step()

            # Update the number counter for seen elements
            how_may_elements_seen += len(labels)
            epoch_iteration += len(labels)

            file_logger.debug(f"In iteration {i} of epoch {epoch} {len(labels)} elements have been seen")
            wandb.log({"Running loss": loss})

            # Check if we should log, based on how many elements we have seen
            if logger.should_log(how_may_elements_seen):

                file_logger.info("Started logging loss information")

                # Log and return loss from training and validation
                logger_metrics = logger.log_process(
                    train_loader,
                    validation_loader,
                    epoch,
                    epoch_iteration
                )

                # Save the metrics returned by the logger
                for metric_key, metric_value in logger_metrics.items():

                    # Check if this is the first time we access this metric
                    # In that case, create at this key a list with that element
                    if training_history.get(metric_key) is None:
                        training_history[metric_key] = [metric_value]

                    # Oterwise, append to the existing list
                    training_history[metric_key].append(metric_value)

            else:
                print(f"[{epoch} / {epoch_iteration}]")

            # Take a snapshot of the network, in case the notebook crashes or we get a timeout
            # TODO -- create a SnapshotTaker class as we have for logs -- snapshot_taker.should_log(i)
            if snapshot_iterations is not None and how_may_elements_seen % snapshot_iterations == 0:

                file_logger.info("Saving a snapshot of the model")

                snapshot_name = "snapshot_" + name + "==" + get_datetime_str()
                snapshot_folder = os.path.join(path, "snapshots")
                filesystem.save_model(net, folder_path = snapshot_folder, file_name = snapshot_name)

    print("Finished training")
    file_logger.info("Finished training")


    # Save the model -- use name + date stamp to save the model
    file_logger.info("Saving the trained model in disk")
    date = get_datetime_str()
    name = name + "==" + date
    filesystem.save_model(net = net, folder_path = path, file_name = name)

    # Return the training hist
    return training_history
