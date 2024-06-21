import random
from typing import List, Tuple

import numpy as np
import torch


class WrappedSubset(torch.utils.data.Dataset):
    """
    Given a subset, adds some attributes that we are using in some places of
    our code. Mainly, adds the `self.targets` attribute

    If underlying dataset has no attribute `targets`, this class will fail to
    initialize
    """

    def __init__(self, subset: torch.utils.data.Subset):
        # Use composition over inheritance. We inherit from Dataset to mark that
        # we are going to implement its interface.
        self.subset = subset

        self.targets = None
        self.__wrap_targets()

    def __wrap_targets(self):
        try:
            self.targets = [
                self.subset.dataset.targets[idx] for idx in self.subset.indices
            ]
        except Exception as e:
            err_msg = f"""Could not wrap subset with additional targets attribute
            Err msg was: {e}"""
            raise ValueError(err_msg)

    # Implement torch.utils.data.Dataset interface using composition
    def __getitem__(self, idx):
        return self.subset.__getitem__(idx)

    def __len__(self):
        return self.subset.__len__()


def split_dataset(
    dataset: torch.utils.data.Dataset, first_element_percentage: float
) -> Tuple[WrappedSubset, WrappedSubset]:
    """
    Given a `dataset`, splits it in two subset. First subset will have
    `first_element_percentage`% elements. Elements are chosen randomly

    We wrap the subsets in our `WrappedSubset` class, so we can access some
    attributes of the normal dataset class, like `self.targets`

    Thus, `dataset` must have `targets` attribute
    """

    # Calculate sizes
    dataset_size = len(dataset)
    first_size = int(dataset_size * first_element_percentage)
    second_size = dataset_size - first_size

    # Split randomly. This gives us a torch.utils.data.Subset class, that need
    # to be transformed to torch.utils.data.Dataset
    first_subset, second_subset = torch.utils.data.random_split(
        dataset, [first_size, second_size]
    )

    # Wrap subset into our WrappedSubset class
    first_dataset = WrappedSubset(first_subset)
    second_dataset = WrappedSubset(second_subset)

    return first_dataset, second_dataset


def split_dataset_disjoint_classes(
    dataset: torch.utils.data.Dataset, first_element_percentage: float
) -> Tuple[WrappedSubset, WrappedSubset]:
    """
    Sample split as `split_dataset`, but with one important property:

    Targets of the two new datasets are disjoint. That is to say, if we found
    one target value in one dataset, we are sure that there is no elemnt in the
    other dataset with the same target

    For example, in our problem, that means that the second dataset only contains
    persons that are not present in the first dataset

    NOTE: due to that behaviour, the obtained percentage is not going to be exact

    NOTE: our algorithm is biased, first dataset tends to be bigger than expected
    while second dataset tends to be smaller
    """

    # Compute the desired min size for the first split
    first_min_size = int(len(dataset) * first_element_percentage)

    # Remove repeated targets
    available_targets = list(set(dataset.targets))

    # We are going to do a lot of searches on the target list
    # So keep a copy of it inside an `np.array` for more speed
    np_targets = np.array(dataset.targets)

    # The indixes of the first dataset
    # We are working with numpy for speed reasons
    first_indixes: np.array = np.array([])

    while len(first_indixes) < first_min_size:
        # Choose a new target and remove it from available targets
        current_target = int(random.choice(available_targets))
        available_targets.remove(current_target)

        # Search for the new indixes
        current_indixes = np.argwhere(np_targets == current_target)

        # Add them to our list
        # We have to check if the list is empty, because concatenating to an
        # empty list returns a list of floats instead of integers :(
        if len(first_indixes) == 0:
            first_indixes = current_indixes
        else:
            first_indixes = np.concatenate((first_indixes, current_indixes), axis=None)

    # Get the indixes of the second dataset
    # Iterate over the available targets (which are targets that have been not
    # used in the first dataset) and get their indixes
    second_indixes = np.array([])
    for current_target in available_targets:
        current_indixes = np.argwhere(np_targets == current_target)

        # Again, same problem with numpy empty lists!
        if len(second_indixes) == 0:
            second_indixes = current_indixes
        else:
            second_indixes = np.concatenate(
                (second_indixes, current_indixes), axis=None
            )

    # Convert both `np.arrays` to normal python lists
    first_indixes = first_indixes.tolist()
    second_indixes = second_indixes.tolist()

    # Split in two datasets using our computed indixes for the datasets
    first_dataset = WrappedSubset(torch.utils.data.Subset(dataset, first_indixes))
    second_dataset = WrappedSubset(torch.utils.data.Subset(dataset, second_indixes))

    return first_dataset, second_dataset
