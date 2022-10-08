import unittest

import torch
import torchvision
import torchvision.transforms as transforms

import src.lib.data_augmentation as data_augmentation

DATA_PATH = "data"
DATASET_PERCENTAGE = 1.0

class TestDataAugmentation(unittest.TestCase):
    """
    Tests are organized in the following way:
        1. `__select_type_of_data_augmentation` gets lazy or normal data augmentation
        2. Each tests has a `__test_<name>` method, that accepts a `lazy` bool to select
           which type of augmentation is going to be used
        3. Then, a `test_<name>` calls `__test_<name>` for each type, removing duplicated test code
           for each type of dataset augmentation
    """

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
            dataset.targets = old_targets[range(0, new_dataset_len)]

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
        self.assertEqual(len(dataset), len(augmented_dataset))

    def test_data_augmentation_with_one_min_img_does_nothing(self, lazy: bool = False):
        self.__test_data_augmentation_with_one_min_img_does_nothing(False)
        self.__test_data_augmentation_with_one_min_img_does_nothing(True)
