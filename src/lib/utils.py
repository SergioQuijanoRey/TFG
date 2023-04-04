"""
Module to put utilities code
"""

from typing import Tuple, List, Dict, Union
import numpy as np
import torch
import os
import wandb
import dotenv

def precompute_dict_of_classes(labels: Union[List[int], np.ndarray]) -> Dict[int, List[int]]:
    """
    Computes a dict containing lists of indixes. Each key of the dict is an int associated to
    the class label. At that position we store a list containing all the indixes associated
    to that class. That's to say, class_positions[i] contains all positions of elements of i-th
    class

    We assume that class are numeric values. That's to say, string classes won't work
    """

    # Init the dict we're going to return
    class_positions = dict()

    # We walk the dataset and assign each element to their position
    for idx, label in enumerate(labels):

        # TODO -- should not be using int(...) to cast torch.Tensor to int
        idx, label = int(idx), int(label)

        # Check if this label has no elements yet
        # In this case, create a list with that index, so later we can append to that list
        if class_positions.get(label) is None:
            class_positions[label] = [idx]
            continue

        # Append the element to this position
        class_positions[label].append(idx)

    return class_positions

def rearrange_indx(first: int, second: int) -> Tuple[int, int]:
    """
    Given two indixes, returns them in a tuple, such that the first element of the tuple is
    less or equal than the second element of the tuple
    """
    if first < second:
        return (first, second)

    return (second, first)

def is_matrix_tensor(tensor: torch.Tensor) -> bool:
    """
    Checks if a given `tensor` is a matrix tensor
    That's to say, if the `tensor` has two modes
    """

    return number_of_modes(tensor) == 2

def is_vector_tensor(tensor: torch.Tensor) -> bool:
    """
    Checks if a given `tensor` is a vector tensor
    That's to say, if the `tensor` has only one mode
    """

    return number_of_modes(tensor) == 1

def number_of_modes(tensor: torch.Tensor) -> int:
    """Given a `tensor`, returns its number of modes"""

    return len(tensor.shape)


def norm_of_each_row(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Given a matrix tensor, containing embedding vectors stored in rows,
    computes the norm of each vector

    Returns a tensor list, with the norm of each row
    """

    # Sanity check
    if is_matrix_tensor(embeddings) is False:
        err_msg = f"""`embeddings` is not a matrix tensor
        Expected order 2, got order {number_of_modes(embeddings)}"""

        raise ValueError(err_msg)

    # We compute the norm of each row of the matrix tensor, using
    # `dim = 1`
    return torch.norm(embeddings, dim = 1)

def change_dir_env_vars(base_path: str):
    """
    Some libraries write to specific dirs. Usually they default to some dir in
    the home folder. In UGR's server, slurm processes don't have access to home

    Thus, here we change that paths using env vars
    """

    # Change dir env vars
    # This way, we write to certain dirs
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(base_path, "wandb_config_dir_testing")
    os.environ["WANDB_CACHE_DIR"] = os.path.join(base_path, "wandb_cache_dir_testing")
    os.environ["WANDB_DIR"] = os.path.join(base_path, "wandb_dir_testing")
    os.environ["WANDB_DATA_DIR"] = os.path.join(base_path, "wandb_datadir_testing")
    os.environ["TORCH_HOME"] = os.path.join(base_path, "torch_home")

def login_wandb():
    """
    Changing some env vars force us to log to wandb again

    We load the API Key from `.env` file that must be present
    """

    dotenv.load_dotenv()
    wandb.login(key = os.environ["WANDB_API_KEY"])
