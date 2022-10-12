"""Different metrics in one place"""

import torch
from torch.utils.data import DataLoader
from torch import mean, nn
import numpy as np
from typing import Callable, Dict, List

import src.lib.core as core
import src.lib.utils as utils
import src.lib.loss_functions as loss_functions

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
def compute_cluster_sizes_metrics(
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

    # Get the portion of the dataset we're interested in
    # Also, use this step to compute the embeddings of the images
    embeddings: torch.Tensor = torch.tensor([]).to(device)
    targets: torch.Tensor = torch.tensor([]).to(device)
    seen_examples = 0

    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Doing this cat is like doing .append() on a python list, but on torch.Tensor, which is
        # much faster. But more important, we can do ".append" in GPU mem without complex conversions
        targets = torch.cat((targets, labels), 0)
        embeddings = torch.cat((embeddings, net(imgs)), 0)

        seen_examples += len(labels)
        if seen_examples >= max_examples:
            break

    # Convert gpu torch.tensor to cpu numpy array
    targets = targets.cpu().numpy()
    embeddings = embeddings.cpu()

    # Pre-compute dict of classes for efficiency
    dict_of_classes = utils.precompute_dict_of_classes(targets)

    # Dict having all the pairwise distances of elements of the same class
    class_distances = compute_intracluster_distances(dict_of_classes, embeddings)

    # With intra-cluster distances, we can compute cluster sizes
    cluster_sizes = [max(class_distance) for class_distance in class_distances.values()]

    # Now, we can compute the three metrics about cluster disntances
    metrics = {
        "min": min(cluster_sizes),
        "max": max(cluster_sizes),
        "mean": np.mean(cluster_sizes),
        "sd": np.std(cluster_sizes),
    }

    return metrics

def compute_intracluster_distances(
    dict_of_classes: Dict[int, List[int]],
    elements: torch.Tensor
) -> Dict[int, List[float]]:
    """
    Aux function that computes, for each cluster, all the distance between two points of that cluster
    """

    class_distances = {label: [] for label in dict_of_classes.keys()}

    # Compute intra-cluster distances
    for curr_class in dict_of_classes.keys():
        for first_indx in dict_of_classes[curr_class]:
            for second_indx in dict_of_classes[curr_class]:

                # We don't want the distance of an element with itself
                # Also, as d(a, b) = d(b, a), we only compute distance for a < b
                # We skip if a > b or a == b, thus, when a >= b
                if first_indx >= second_indx:
                    continue

                # Get the distance and append to the list
                distance = loss_functions.distance_function(elements[first_indx], elements[second_indx])
                class_distances[curr_class].append(float(distance))

    # Some classes can have only one element, and thus no class distance
    # We don't consider this classes
    class_distances = {
        label: class_distances[label]
        for label in class_distances.keys()
        if len(class_distances[label]) > 0
    }

    return class_distances
