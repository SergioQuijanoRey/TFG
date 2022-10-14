"""
Module to put utilities code
"""

from typing import Tuple, List, Dict, Union
import numpy as np

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

