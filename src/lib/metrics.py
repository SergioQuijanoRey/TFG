"""Different metrics in one place"""

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from typing import Callable, Dict, List, Tuple
import itertools

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

# TODO -- PERF -- this function is taking a lot of time
# TODO -- PERF -- but most of the time is spent getting the data
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

# TODO -- PERF -- This function is taking a lot of cumulative time
def __get_portion_of_dataset_and_embed(
    data_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    max_examples: int,
    fast_implementation: bool = False
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    This aux function gets a portion of a dataset. At the same time, it gets the embedding of the
    images we're interested in, through given net

    We are getting the first `max_examples` elements of the data. No random
    sampling or other technique is done

    `fast_implementation` defines wether or not use the fast implementation.
    That fast implementation makes a better use of device memory (CPU or GPU)
    but needs more resources. Big data batches can crash the running program, so
    use it carefully. That's why default value is False
    """

    # Fast implementation
    if fast_implementation is True:
        return __get_portion_of_dataset_and_embed_fast(
            data_loader,
            net,
            max_examples,
        )

    # Slow implementation
    return __get_portion_of_dataset_and_embed_slow(
        data_loader,
        net,
        max_examples,
    )

def __get_portion_of_dataset_and_embed_fast(
    data_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    max_examples: int,
):
    """
    Fast implementation of `__get_portion_of_dataset_and_embed`

    Be cautious of resource consumption
    """

    # Get memory device we are using
    device = core.get_device()

    # Get all the batches in the data loader
    # This is the most slow part of the function
    elements = [
        (imgs.to(device), labels.to(device))
        for (index, (imgs, labels)) in enumerate(data_loader)
        if index < max_examples
    ]

    # Now, split previous list of pairs to a pair of lists
    imgs, targets = zip(*elements)

    # imgs, targets are a list of lists, because dataloader returns elements in
    # batches. We could flatten now the two lists of lists, but the network
    # expects data in batches. So we use that structure to pass the images
    # through the network
    #
    # This produces a list of tensors containing embeddings batches
    # Further transformation needs to be done
    embeddings = [net(img_batch) for img_batch in imgs]

    # Now, flatten the targets
    targets = list(itertools.chain(*targets))

    # Convert to numpy array
    # Numpy arrays cannot live in GPU memory
    # Also, we have a python list of tensors
    targets = np.array([
        target.cpu()
        for target in targets
    ])

    # Make the conversion for the embeddings
    # We have a list of tensors. For computing its distance, we need a matrix
    # tensor. Also, embeddings is a list of matrix tensors. `torch.stack` would
    # produce a tensor with a list of matrix tensors. We want a single matrix
    # tensor, thus, we must use `torch.cat`
    #
    # However, we can also have a list of vector tensors. This happens when
    # we use a batch size of one. So we look to the first entry in the embeddings
    # list and act accordingly
    #
    # Both conversion methods are pytorch calls, that should be fast
    conversion_method = None
    if utils.is_vector_tensor(embeddings[0]):
        conversion_method = lambda embeddings: torch.stack(embeddings)
    elif utils.is_matrix_tensor(embeddings[1]):
        conversion_method = lambda embeddings: torch.cat(embeddings, dim = 0)
    else:

        # Raise an informative exception
        err_msg = f"""`embeddings` has {utils.number_of_modes(embeddings)}
        We don't have a conversion method to get the correct format for \
        computing the distance dict"""

        raise Exception(err_msg)

    # Now, use the conversion method to get the correct format
    embeddings = conversion_method(embeddings)

    # Check that the embeddings tensor is in fact a matrix tensor
    # That's to say, that it has two modes
    if utils.is_matrix_tensor(embeddings) is False:

        err_msg = f"""Embeddings should be a matrix tensor or a vector matrix
        Embeddings tensor has {len(embeddings.shape)} modes, instead of two or one
        """

        raise Exception(err_msg)

    return embeddings, targets

def __get_portion_of_dataset_and_embed_slow(
    data_loader: torch.utils.data.DataLoader,
    net: torch.nn.Module,
    max_examples: int,
):
    """Slow implementation for __get_portion_of_dataset_and_embed"""

    # Get memory device we are using
    device = core.get_device()

    embeddings: torch.Tensor = torch.tensor([]).to(device)
    targets: torch.Tensor = torch.tensor([]).to(device)
    seen_examples = 0

    for imgs, labels in data_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # Doing this cat is like doing .append() on a python list, but on torch.Tensor, which is
        # much faster. But more important, we can do ".append" in GPU mem without complex conversions
        targets = torch.cat((targets, labels), 0)
        embeddings = torch.cat((embeddings, net(imgs).to(device)), 0)

        seen_examples += len(labels)
        if seen_examples >= max_examples:
            break

    # Convert gpu torch.tensor to cpu numpy array
    targets = targets.cpu().numpy()
    embeddings = embeddings.cpu()

    return embeddings, targets

# TODO -- PERF -- this function is taking a lot of time
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
    2. Mean cluster distance over all clusters
    4. SDev of cluster distances

    Given a cluster, its distance is defined as the max distance between two points of that cluster

    """

    # Move network to proper device
    device = core.get_device()
    net.to(device)

    # Get the portion of the dataset we're interested in
    # Also, use this step to compute the embeddings of the images
    embeddings, targets = __get_portion_of_dataset_and_embed(data_loader, net, max_examples)

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
        "mean": float(np.mean(cluster_sizes)),
        "sd": float(np.std(cluster_sizes)),
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

    # Precompute all pairwise distances
    base = loss_functions.BatchBaseTripletLoss()
    pairwise_distances = base.precompute_pairwise_distances(elements)

    # Compute intra-cluster distances
    for curr_class in dict_of_classes.keys():
        for first_indx in dict_of_classes[curr_class]:
            for second_indx in dict_of_classes[curr_class]:

                # We don't want the distance of an element with itself
                # Also, as d(a, b) = d(b, a), we only compute distance for a < b
                # We skip if a > b or a == b, thus, when a >= b
                if first_indx >= second_indx:
                    continue

                # Use the precomputation of all pairwise distances
                class_distances[curr_class].append(
                    float(pairwise_distances[(first_indx, second_indx)])
                )

    # Some classes can have only one element, and thus no class distance
    # We don't consider this classes
    class_distances = {
        label: class_distances[label]
        for label in class_distances.keys()
        if len(class_distances[label]) > 0
    }

    return class_distances

# TODO -- PERF -- this takes too much time to compute
def compute_intercluster_metrics(
        data_loader: torch.utils.data.DataLoader,
        net: torch.nn.Module,
        max_examples: int
    ) -> Dict[str, float]:
    """
    Computes metrics about intercluster metrics
    This information will be:

    1. Max intercluster distance over all clusters
    2. Min intercluster distance over all clusters
    3. Mean intercluster distance
    4. SDev of intercluster distances

    Given two clusters, its distance is defined as the min distance between two points, one from
    each cluster
    """

    # Move network to proper device
    device = core.get_device()
    net.to(device)

    # Get the portion of the dataset we're interested in
    # Also, use this step to compute the embeddings of the images
    # TODO -- PERF -- this is taking too much time
    embeddings, targets = __get_portion_of_dataset_and_embed(data_loader, net, max_examples)

    # Pre-compute dict of classes for efficiency
    dict_of_classes = utils.precompute_dict_of_classes(targets)

    # Precompute pairwise distances for efficiency
    distances = __compute_pairwise_distances(embeddings)

    # Now compute inter distances
    intercluster_distances: Dict[Tuple[int, int], float] = compute_intercluster_distances(distances, dict_of_classes)

    # Flatten prev dict, indexed by two indixes
    flatten_intercluster_distances = intercluster_distances.values()

    # Convert the data to a tensor containing all distances
    flatten_intercluster_distances = torch.stack(list(flatten_intercluster_distances))

    # Now we can easily return the metrics
    metrics = {
        "min": float(min(flatten_intercluster_distances)),
        "max": float(max(flatten_intercluster_distances)),
        "mean": float(torch.mean(flatten_intercluster_distances)),
        "sd": float(torch.std(flatten_intercluster_distances, unbiased = False)),
    }

    return metrics

def compute_intercluster_distances(
    distances: torch.Tensor,
    dict_of_classes: Dict[int, List[int]]
) -> Dict[Tuple[int, int], float]:
    """
    Computes the intercluster distances of a given set of embedding clusters

    As D(cluster_a, cluster_b) = D(cluster_b, cluster_a), we only compute these distances for a < b
    Also, we are not interested in case a = b
    """

    # Generate the list of pairs of clusters we are going to explore
    cluster_pairs = [
        (first, second)
        for first in dict_of_classes.keys()
        for second in dict_of_classes.keys()
        if first < second
    ]

    # For each cluster pair, compute its intercluster distance
    intercluster_distance = dict()
    for (first_cluster, second_cluster) in cluster_pairs:

        intercluster_distances = [
            # Distances have only indixes (first, second) such that first < second
            # So we rearrange them for assuring this condition
            distances[utils.rearrange_indx(first_indx, second_indx)]
            for first_indx in dict_of_classes[first_cluster]
            for second_indx in dict_of_classes[second_cluster]
        ]

        intercluster_distance[(first_cluster, second_cluster)] = min(intercluster_distances)

    return intercluster_distance

def __compute_pairwise_distances(embeddings: torch.Tensor) -> Dict[Tuple[int, int], float]:
    """
    Computes the pairwise distance of elements in the embeddings set. Only for elements indexed a, b
    such that a < b
    """

    # Use BatchBaseTripletLoss class for doing the computation
    base_loss = loss_functions.BatchBaseTripletLoss()
    distances = base_loss.precompute_pairwise_distances(embeddings)

    # Prev dict has elements [a, a] with distance 0. We are not interested in those
    distances = {index: distances[index] for index in distances.keys() if index[0] != index[1]}

    return distances
