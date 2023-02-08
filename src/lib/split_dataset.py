import torch
from typing import Tuple

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
                self.subset.dataset.targets[idx]
                for idx in self.subset.indices
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
    dataset: torch.utils.data.Dataset,
    first_element_percentage: float
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
        dataset,
        [first_size, second_size]
    )

    # Wrap subset into our WrappedSubset class
    first_dataset = WrappedSubset(first_subset)
    second_dataset = WrappedSubset(second_subset)

    return first_dataset, second_dataset
