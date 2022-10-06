"""
Module to put utilities code
"""

from typing import List, Dict


def precompute_dict_of_classes(labels: List[int]) -> Dict[int, List[int]]:
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

        # Check if this label has no elements yet
        # In this case, create a list with that index, so later we can append to that list
        if class_positions.get(label) is None:
            class_positions[label] = [idx]
            continue

        # Append the element to this position
        class_positions[label].append(idx)

    return class_positions
