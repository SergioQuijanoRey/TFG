"""Different metrics in one place"""

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np

from typing import Callable, Dict

import core
import utils
import loss_functions

def calculate_mean_loss(net: nn.Module, data_loader: DataLoader, max_examples: int, loss_function) -> float:
    """
    Calculates mean loss over a data set

    Parameters:
    ===========
    net: the net we are using to calculate metrics
    data_loader: wraps the dataset (training / validation set)
    max_examples: in order to not have to iterate over the whole set (slow computation), it
                  specifies max number of examples to see
    loss_function: the loss function we're using to compute the metric

    Returns:
    =========
    mean_loss: the mean of the loss over seen examples
    """

    curr_examples = 0
    acumulated_loss = 0.0
    for data in data_loader:
        # Unwrap input / outputs
        inputs, labels = core.unwrap_data(data)

        # Calculate outputs
        outputs = net(inputs)

        # Calculate loss
        acumulated_loss += loss_function(outputs, labels)

        # Update seen examples and check for stop condition
        curr_examples += inputs.size(0)
        if curr_examples >= max_examples:
            break

    mean_loss = acumulated_loss / curr_examples
    return mean_loss

def calculate_accuracy(net: nn.Module, data_loader: DataLoader, max_examples: int) -> float:
    """
    Calculates accuracy over a data set

    Parameters:
    ===========
    net: the net we are using to calculate metrics
    data_loader: wraps the dataset (training / validation set)
    max_examples: in order to not have to iterate over the whole set (slow computation), it
                  specifies max number of examples to see

    Returns:
    =========
    accuracy: percentage of well-predicted examples
              percentage in range [0, 100]
    """

    curr_examples = 0
    correct_predictions = 0.0
    for data in data_loader:
        # Unwrap input / outputs
        inputs, labels = core.unwrap_data(data)

        # Calculate outputs
        outputs = net(inputs)
        _, outputs = torch.max(outputs.data, 1)

        # Calculate correct predictions
        correct_predictions += (outputs == labels).sum().item()

        # Update seen examples and check for stop condition
        curr_examples += inputs.size(0)

        if curr_examples >= max_examples:
            break

    acc = correct_predictions / curr_examples * 100.0
    return acc

def calculate_mean_triplet_loss_offline(net: nn.Module, data_loader: DataLoader, loss_function) -> float:
    """
    Calculates mean loss over a data set, for a triplet-like loss
    Offline version

    @param net: the net we are using to calculate metrics
    @param data_loader: wraps the dataset (training / validation set)
    @param max_examples: in order to not have to iterate over the whole set (slow computation), it
                         specifies max number of examples to see
    @param loss_function: the loss function we're using to compute the metric. Has to be triplet-like loss
    @param online: controls the use of minibatches

    @returns mean_loss: the mean of the loss over seen examples
    """

    # Get device where we are training
    device = core.get_device()

    # Calculate loss in the given dataset
    acumulated_loss = 0.0
    for data in data_loader:

        # Calculate embeddings
        # Put them together in one batch
        batch = [net(item[None, ...].to(device)) for item in data]

        # Calculate loss
        acumulated_loss += loss_function(batch)


    mean_loss = acumulated_loss / len(data_loader.dataset)
    return mean_loss

def calculate_mean_loss_function_online(
    net: nn.Module,
    data_loader: DataLoader,
    loss_function: Callable,
    max_examples: int,
    greater_than_zero: bool = False
) -> float:
    """
    Calculates mean loss over a data set, for a triplet-like loss online version
    The loss function is determined as a parameter of the function

    @param net: the net we are using to calculate metrics
    @param data_loader: wraps the dataset (training / validation set)
    @param max_examples: in order to not have to iterate over the whole set (slow computation), it
                         specifies max number of examples to see
    @param loss_function: the loss function we're using to compute the metric. Has to be triplet-like loss
    @param max_examples: max examples to evaluate in order to compute the metric
    @param greater_than_zero: choose if we want to use only greater than zero values for computing
                              the mean loss

    @returns mean_loss: the mean of the loss over seen examples
    """

    # Get device where we are training
    device = core.get_device()

    # Calculate loss in the given dataset
    acumulated_loss = 0.0
    curr_examples = 0
    gt_than_zero_examples = 0
    for data in data_loader:

        # Unwrap the data
        imgs, labels = data

        # Calculate embeddings
        embeddings = net(imgs.to(device))

        # Calculate loss
        current_loss = loss_function(embeddings, labels)
        acumulated_loss += current_loss

        # Update seen examples and check for break condition
        curr_examples += imgs.size(0)
        if curr_examples >= max_examples:
            break

        if current_loss > 0:
            gt_than_zero_examples += imgs.size(0)


    # Compute mean loss in function of gt_than_zero_examples
    denominator = gt_than_zero_examples if greater_than_zero else curr_examples
    mean_loss = acumulated_loss / denominator

    return mean_loss

# TODO -- TEST -- write tests for this function
def compute_cluster_sizes(
        data_loader: torch.utils.data.DataLoader,
        net: torch.nn.Module,
        max_examples: int
    ) -> Dict[str, float]:
    """
    Computes metrics about cluster sizes
    This information will be:

    1. Max cluster distance over all clusters
    2. Min cluster distance over all clusters
    3. SDev of cluster distances

    Given a cluster, its distance is defined as the max distance between two points of that cluster

    """

    # Move network to proper device
    device = core.get_device()
    net.to(device)

    # We need information about the dataset
    # For this metric, the way we sample the data should not be important
    dataset = data_loader.dataset

    # Get the portion of the dataset we're interested in
    # Also, use this step to compute the embeddings of the images
    embeddings = [
        net(
            # .to(device) because we need to have data in proper memory
            # .unsqueeze(0) because the net is expecting batches of images
            img.unsqueeze(0).to(device)
        )
        for indx, (img, _) in enumerate(dataset)
        if indx < max_examples
    ]

    # Pre-compute dict of classes for efficiency
    dict_of_classes = utils.precompute_dict_of_classes(dataset.targets[:max_examples])

    # Dict having all the pairwise distances of elements of the same class
    class_distances = {label: [] for label in dict_of_classes.keys()}

    # Compute intra-cluster distances
    # TODO -- move to other aux function
    for curr_class in dict_of_classes.keys():
        for first_indx in dict_of_classes[curr_class]:
            for second_indx in dict_of_classes[curr_class]:

                # We don't want the distance of an element with itself
                if first_indx == second_indx:
                    continue

                # Get the distance and append to the list
                distance = loss_functions.distance_function(embeddings[first_indx], embeddings[second_indx])
                class_distances[curr_class].append(distance)


    # With intra-cluster distances, we can compute cluster sizes
    cluster_sizes = [
        max(distance)
        for distance in class_distances[curr_class]
        for curr_class in dict_of_classes.keys()
    ]

    # Now, we can compute the three metrics about cluster disntances
    metrics = {
        "min": min(cluster_sizes),
        "max": max(cluster_sizes),
        "sd": np.std(cluster_sizes),
    }

    return metrics
