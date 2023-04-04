import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import ShuffleSplit

import src.lib.metrics as metrics

from typing import List, Callable

def custom_cross_validation(
    train_dataset: Dataset,
    k: int,
    random_seed: int,
    network_trainer: Callable[[DataLoader], torch.nn.Module],
    loader_generator: Callable[[Dataset], DataLoader],
    loss_function: Callable[[torch.nn.Module, DataLoader], float]
):
    """
    Perform k-fold cross validation

    `train_dataset`: dataset where we perform k-fold cross validation

    `k`: number of folds that we want to use

    `random_seed`: seed, for reproducibility purposes

    `network_trainer` should be a function that produces a network trained on
    a given dataloader for each fold (see `loader_generator`). It has to return
    the trained network

    `loader_generator` should be a function that takes one fold (which is a
    dataset) and converts it to a `DataLoader`. For example, taking care of
    creating one of our custom dataloaders

    `loss_function` should be a function that takes a trained network and
    the validation fold dataloader, and produces the loss value that we want
    to optimize
    """

    # Object to generate the folds in a easy way
    ss = ShuffleSplit(n_splits=k, test_size=0.25, random_state=random_seed)

    # List where we're going to store all the loss values for each fold
    losses: List[float] = []

    # Iterate over each fold
    for train_index, validation_index in ss.split(train_dataset):

        # We have the index of the elements for training and validation for this
        # fold. So we have to take that elements and create a dataset for each
        train_fold = [train_dataset[idx] for idx in train_index]
        validation_fold = [train_dataset[idx] for idx in validation_index]

        # Transform dataset folds to dataloaders
        train_loader = loader_generator(train_fold)
        validation_loader = loader_generator(validation_fold)

        # Train the network and get the trained network
        net = network_trainer(train_loader)

        # Evaluate  the network on the validation fold
        net.eval()
        loss = loss_function(net, validation_loader)

        # Add that loss to our loss list
        losses.append(loss)

    # Use numpy instead of vanilla lists
    return np.array(losses)
