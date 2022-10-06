"""
Module where custom samplers go
"""

import torch
import random
import src.lib.utils as utils

from typing import Iterator, List, Dict, Optional

import logging
file_logger = logging.getLogger("MAIN_LOGGER")

class CustomSampler(torch.utils.data.Sampler):
    """
    Custom sampler that implements the sampling explained in the reference paper

    In this sampling, in order to generate each minibatch, the following is done:
        1. Sample randomly P classes from all available classes
        2. For each class, sample randomly K elements

    Also, in the paper, they suggest that P and K should be chosen such as
    P * K \\approx 3 * n ; beign n an integer. That suggestion comes from experimentation

    Used this tutorial to code this class:
    https://www.scottcondron.com/jupyter/visualisation/audio/2020/12/02/dataloaders-samplers-collate.html
    """

    def __init__(self, P: int, K: int, dataset: torch.utils.data.Dataset):
        self.P = P
        self.K = K
        self.dataset = dataset
        self.labels = self.dataset.targets

        # We precomputate a dict having lists of indixes, to speed things up
        # Also, this dict is not freezed, we remove elements of it as we add them
        # to the index sequence
        # At each new epoch, this dict is generated again
        # `self.dict_of_classes[4]` has all the indixes corresponding to elements with class 4
        # Thus, `self.dict_of_classes[4][0]` has the index of the first element with class 4
        # TODO -- this should be "private", thats `_dict_of_classes`
        self.dict_of_classes: Optional[Dict[int, List[int]]] = None

        # We are going to build a list of indexes that we iter sequentially
        # Each epoch we generate a new random sequence respecting P - K sampling
        # This is the list that we return in __iter__ method
        # TODO -- this should be private, thats `_index_list`
        self.index_list: Optional[List[int]] = None

        # Because of P-K sampling, with random choices, each sampling has a different size
        # Thus, this value is changed every time we generate a new sampling
        self.len: Optional[int] = None

        # Some methods need to iterate over all possible values of classes
        # So we should compute this list of classes
        # We are assuming that targets are numeric values
        self.classes: List[int] = self.__compute_dict_of_classes(
            # This if else expression makes sure that the method gets a torch.Tensor
            # Depending on the dataset, we can have self.labels to be a List[int]
            torch.Tensor(self.labels) if type(self.labels) is list else self.labels
        )

    def __iter__(self) -> Iterator:

        # Generate random index list
        self.index_list = self.generate_index_sequence()

        # Return iterator to that index list
        return iter(self.index_list)


    def __len__(self) -> int:
        """
        Len of the __iter__ generated in this class.

        As we are doing P-K sampling with random choices, thus some elements of the dataset are
        going to not be sampled. This method computes how many elements are going to be sampled

        Thus, this inequality holds:
            self.__len__() <= self.dataset.__len()__

        # TODO -- add two methods, for lower and upper boundaries in __len__
                  This way, if this exception is raised from pytorch, we can fix it somehow
        """

        # Raising an error if `self.len` is None to avoid problems with pytorch implementations
        # If this problem arises, we will take care of it later
        if self.len is None:
            raise Exception("CustomSampler.__len__ has not been computed yet!")

        return self.len

    def generate_index_sequence(self) -> List[int]:
        """
        Generates the sequence of indixes that we are going to return in __iter__

        We have to make that sequence such that P-K sampling is done when getting minibatches of
        size P*K (each minibatch has P random classes, K random elements for each class)
        """

        # Index list that we are going to return
        # Is a flat list but verifying that each minibatch of size P*K has been constructed the
        # way we specified before
        list_of_indixes: List[int] = []

        # Re-generate the list of indixes splitted by class
        self.dict_of_classes = None
        self.dict_of_classes = utils.precompute_dict_of_classes(self.labels)

        # Classes that have more than `self.K` elements, so can be used for sampling
        # We start with all classes and make an starting clean (because certain class can have
        # less than `self.K` elements in the start)
        available_classes = self.classes
        available_classes = self.remove_empty_classes(available_classes)

        # Log and do some debugging
        # If first clean lets us with less than `self.P` classes, we have a problem
        file_logger.debug(f"After first cleaning, there are {len(available_classes)} available_classes")
        if len(available_classes) < self.P:
            err_msg = f"After first cleaning, we have less than `self.P` = {self.P} classes"
            file_logger.error(err_msg)
            raise Exception(err_msg)

        # Make minibatches while there are at least `self.P` classes with at least `self.K` elements
        # each class
        while len(available_classes) >= self.P:

            # Choose the P classes used in this iteration
            random.shuffle(available_classes)
            curr_classes = available_classes[:self.P]

            # Generate new batch and add to the list of indixes
            # We use + operator because we want a plain list, not a list of list
            # (list of minibatches)
            list_of_indixes = list(list_of_indixes) + list(self.__new_batch(curr_classes))

            # Check for classes that has less than self.K images available
            available_classes = self.remove_empty_classes(available_classes)

        # Before returning the list, change `self.len`
        self.len = len(list_of_indixes)

        # Log some info about the process
        file_logger.debug(f"Generated `list_of_indixes` is {list_of_indixes}")

        # Make some debugging
        if self.len == 0:
            err_msg = "CustomSampler.generate_index_sequence has produced and empty sequence of index to iterate"
            file_logger.error(err_msg)
            raise Exception(err_msg)

        return list_of_indixes

    def remove_empty_classes(self, class_list: List[int]) -> List[int]:
        """
        Remove classes from a given list of classes that have less than `self.K` elements
        """

        return [
            curr_class for curr_class in class_list
            if
                # We remove classes that have less than self.K elements
                # But some classes could not be in the dict, so check first if they're stored
                # in the dict first, to avouid `KeyError 0`
                self.dict_of_classes.get(curr_class) is not None and
                len(self.dict_of_classes[curr_class]) >= self.K
        ]


    def __new_batch(self, classes: List[int]) -> List[int]:
        """
        Given a list of `self.P` classes, picks `self.K` elements for each class. Thus, working the
        K in P-K sampling (because P classes already picked)
        """

        # Check that `self.P` classes are picked
        # TODO -- remove this in the future, because is checked in unit tests
        assert self.P == len(classes), f"We have {len(classes)} classes, when P = {self.P}"

        batch = []

        for curr_class in classes:
            for _ in range(self.K):

                # Choose a random image of this class
                curr_idx_position = random.randint(0, len(self.dict_of_classes[curr_class]) - 1)
                curr_idx = self.dict_of_classes[curr_class][curr_idx_position]

                # Then, this image is no longer available
                del self.dict_of_classes[curr_class][curr_idx_position]

                # Add chosen image to the batch
                batch.append(curr_idx)

        return batch

    def __compute_dict_of_classes(self, labels: torch.Tensor) -> List[int]:
        """
        Given a tensor containing all labels, computes a list with the classes. That's to say,
        a list with the unique values of the tensor

        For example, given torch.Tensor([1, 1, 2, 3, 1, 4]), [1, 2, 3, 4] should be returned

        We are also sorting the classes, but this is not need for most of the methods

        TODO -- maybe is repeated in BaseTripletLoss, so might need a refactor. Not quite sure
                about this
        """

        # Use torch func to get the list of classes
        unique_labels: torch.Tensor = torch.unique(labels)

        # We want to work with vanilla python list
        unique_labels: List[int] = list(unique_labels)

        # Even though we make the annotation that `unique_labels` is `List[int]`, it can be a
        # `List[torch.Tensor]`, so with this check we make sure that we have a `List[int]`
        if type(unique_labels[0]) is torch.Tensor:
            unique_labels = [int(x) for x in unique_labels]

        return unique_labels


