import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.model_selection import ShuffleSplit
from enum import Enum

import src.lib.metrics as metrics
from src.lib.split_dataset import WrappedSubset

from typing import List, Callable


class FoldType(Enum):
    """
    Enumerate to define which fold are we dealing with
    Some functions might behave different if we are in one fold or another
    """
    TRAIN_FOLD = "Train Fold"
    VALIDATION_FOLD = "Validation Fold"

def custom_cross_validation(
    train_dataset: Dataset,
    k: int,
    random_seed: int,
    network_creator: Callable[[], torch.nn.Module],
    network_trainer: Callable[[DataLoader, torch.nn.Module], torch.nn.Module],
    loader_generator: Callable[[Dataset, FoldType], DataLoader],
    loss_function: Callable[[torch.nn.Module, DataLoader], float]
) -> np.ndarray:
    """
    Perform k-fold cross validation

    `train_dataset`: dataset where we perform k-fold cross validation

    `k`: number of folds that we want to use

    `random_seed`: seed, for reproducibility purposes

    `network_creator` should be a function that creates a new and untrained
    network, that later we are going to use in the training process

    `network_trainer` should be a function that, given a network and a dataloader,
    trains the network and returns that trained network

    `loader_generator` should be a function that takes one fold (which is a
    dataset) and their type (training or validation), and converts it to a `DataLoader`.
    For example, taking care of creating one of our custom dataloaders if we
    are in the training fold, and using a normal dataloader for the validation
    fold

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

        # This transformation avoids using np.int to index a dataset
        train_index = [int(idx) for idx in train_index]
        validation_index = [int(idx) for idx in validation_index]

        # We have the index of the elements for training and validation for this
        # fold. So we have to take that elements and create a dataset for each
        # We use our WrappedSubset class to avoid the problems that Subset provokes
        train_fold = WrappedSubset(Subset(train_dataset, train_index))
        validation_fold = WrappedSubset(Subset(train_dataset, validation_index))


        # Transform dataset folds to dataloaders
        train_loader = loader_generator(train_fold, FoldType.TRAIN_FOLD)
        validation_loader = loader_generator(validation_fold, FoldType.VALIDATION_FOLD)

        # Generate a network, train it and get the trained network
        # Training can fail (i.e. backward losses can be None if parameters are
        # really bad)
        net = network_creator()
        try:
            net = network_trainer(train_loader, net)
        except Exception as e:
            print("Failed training in one of the k-folds ")
            raise e

        # This is useful for trying to avoid memory issues
        del train_fold
        del train_index
        del train_loader
        torch.cuda.empty_cache()

        # Evaluate  the network on the validation fold
        net.eval()
        loss: torch.Tensor = loss_function(net, validation_loader)

        # Before appending this loss to a list, we must detach the tensor from
        # the computation graph. Otherwise we will run out of memory
        # This was indicated at:
        # https://discuss.pytorch.org/t/memory-management-using-pytorch-cuda-alloc-conf/157850/8
        loss: float = float(loss.detach())
        losses.append(loss)
        torch.cuda.empty_cache() # Trying to avoid mem issues

        # k-fold cross validation can be very slow
        # So this logs are very important to watch the process
        print(f"🔎 Current fold loss is {loss}")

    # Use numpy instead of vanilla lists
    return np.array(losses)
