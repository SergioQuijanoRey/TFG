import unittest

import torch
import torchvision
import torchvision.transforms as transforms
from collections import Counter
from functools import lru_cache


import src.lib.data_augmentation as data_augmentation

DATA_PATH = "data"
DATASET_PERCENTAGE = 0.1

class TestDataAugmentation(unittest.TestCase):
    """
    Tests are organized in the following way:
        1. `__select_type_of_data_augmentation` gets lazy or normal data augmentation
        2. Each tests has a `__test_<name>` method, that accepts a `lazy` bool to select
           which type of augmentation is going to be used
        3. Then, a `test_<name>` calls `__test_<name>` for each type, removing duplicated test code
           for each type of dataset augmentation
    """

    # Cache so we don't download many times the same dataset
    @lru_cache(maxsize = 32)
    def __load_LFW_dataset(self, percentage: float = 1.0):
        """
        Aux function to load LFW into a torch.Dataset
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = torchvision.datasets.LFWPeople(
            root = DATA_PATH,
            split = "train",
            download = True,
            transform = transform,
        )


        # Get a portion of the dataset if percentage is lower than 1
        if percentage < 1:

            # Subset class don't preserve targets, so we keep this targets to add them later
            old_targets = dataset.targets

            # We don't care about shuffling, this is only for speeding up some computations on
            # unit tests, and they don't rely on shuffling
            new_dataset_len = int(percentage * len(dataset))
            dataset = torch.utils.data.Subset(dataset, range(0, new_dataset_len))

            # Add the prev targets to the dataset manually
            dataset.targets = old_targets[0:new_dataset_len]

        return dataset

    def __select_type_of_data_augmentation(self, lazy: bool = False):
        """Selects wether or not use lazy augmentation """

        return data_augmentation.AugmentatedDataset if lazy is False else data_augmentation.LazyAugmentatedDataset


    def __test_data_augmentation_with_one_min_img_does_nothing(self, lazy: bool = False):

        # Get the type of augmentation and the dataset
        Augmentation = self.__select_type_of_data_augmentation(lazy)
        dataset = self.__load_LFW_dataset(DATASET_PERCENTAGE)

        # Augmentate the dataset using `min_number_of_images = 1`
        augmented_dataset = Augmentation(
            base_dataset = dataset,
            min_number_of_images = 1,
            transform = lambda x: x
        )

        # Check that the sizes of the images are the same
        self.assertEqual(len(dataset), len(augmented_dataset), "Augmented dataset should not have new images")

    def test_data_augmentation_with_one_min_img_does_nothing(self, lazy: bool = False):
        self.__test_data_augmentation_with_one_min_img_does_nothing(False)
        self.__test_data_augmentation_with_one_min_img_does_nothing(True)

    def __test_original_dataset_is_not_modified(self, lazy: bool = False):

        # Get the type of augmentation and the dataset
        Augmentation = self.__select_type_of_data_augmentation(lazy)
        dataset = self.__load_LFW_dataset(DATASET_PERCENTAGE)

        # Some stats to see if the original dataset mutates
        original_len = len(dataset)

        # Perform some data augmentation
        augmented_dataset = Augmentation(
            base_dataset = dataset,
            min_number_of_images = 4,
            transform = lambda x: x
        )

        # Check that original dataset is not modified
        self.assertEqual(len(dataset), original_len, "Original dataset should not have new images, should not mutate")

    def test_original_dataset_is_not_modified(self):
        self.__test_original_dataset_is_not_modified(False)
        self.__test_original_dataset_is_not_modified(True)

    def __test_that_all_classes_have_at_least_min_images(self, lazy: bool = False):

        # Get the type of augmentation and the dataset
        Augmentation = self.__select_type_of_data_augmentation(lazy)
        dataset = self.__load_LFW_dataset(DATASET_PERCENTAGE)

        # Perform some data augmentation
        K = 4
        augmented_dataset = Augmentation(
            base_dataset = dataset,
            min_number_of_images = K,
            transform = lambda x: x
        )

        # Check that the number of classes that have at least `min_number_of_images` is 100%
        # TODO -- this code is repeated from the Notebook, function `how_many_classes_have_at_least_K_image`
        how_many_images_per_class = Counter(augmented_dataset.targets)
        classes_with_at_least_K_images = [
            curr_class
            for curr_class, curr_value in how_many_images_per_class.items()
            if curr_value >= K
        ]
        n = len(classes_with_at_least_K_images)
        self.assertEqual(n, len(how_many_images_per_class), f"All classes must have at least K = {K} images")

    def test_that_all_classes_have_at_least_min_images(self):

        self.__test_that_all_classes_have_at_least_min_images(False)
        self.__test_that_all_classes_have_at_least_min_images(True)

    def __test_augmented_dataset_is_bigger_that_original_dataset(self, lazy: bool):

        # Get the type of augmentation and the dataset
        Augmentation = self.__select_type_of_data_augmentation(lazy)
        dataset = self.__load_LFW_dataset(DATASET_PERCENTAGE)

        # Perform some augmentation
        augmented_dataset = Augmentation(
            base_dataset = dataset,
            min_number_of_images = 3,
            transform = lambda x: x
        )

        # Check that the size of the dataset has grown
        self.assertGreater(len(augmented_dataset), len(dataset))

    def test_augmented_dataset_is_bigger_that_original_dataset(self):

        self.__test_augmented_dataset_is_bigger_that_original_dataset(False)
        self.__test_augmented_dataset_is_bigger_that_original_dataset(True)
