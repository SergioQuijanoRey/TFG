import unittest

import torch
import torchvision
import torchvision.transforms as transforms
from collections import Counter

from src.lib.sampler import CustomSampler

# Parameters for this tests
#===================================================================================================
DATA_PATH = "data"
BATCH_SIZE = 32
NUM_WORKERS = 1
DATASET_PERCENTAGE = 0.1

class TestCustomSampler(unittest.TestCase):
    """
    Tests for the class `CustomSampler`
    Tests are organized in the following way:
        1. `__get_dataset` can return either MNIST or LFW dataset
        2. Each tests has a `__test_<name>` method, that accepts a `dataset_selection` string to select
           which dataset is going to be used
        3. Then, a `test_<name>` calls `__test_<name>` for each dataset, removing duplicated test code
           for each dataset
    """

    def __load_MNIST_dataset(self, percentage: float = 1.0):
        """
        Aux function to load MNIST into a torch.Dataset

        Parameters:
        ===========
        percentage: percentage of the dataset we want to get
                    Use lower percentages for faster checks
        """

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = torchvision.datasets.MNIST(
            root = DATA_PATH,
            train = True,
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

    def __load_LFW_dataset(self):
        """
        Aux function to load LFW into a torch.Dataset

        Parameters:
        ===========
        percentage: percentage of the dataset we want to get
                    Use lower percentages for faster checks
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

        # For this dataset, we have so many classes with only one image that we must use the whole
        # dataset. So make percentage always be one

        return dataset

    def __get_dataset(self, dataset: str = "MNIST") -> torch.utils.data.Dataset:
        """
        This function returns one of the two datasets we're using for testing
        @param dataset: string that can be `"MNIST"` or `"LFW"`
        @returns specified dataset
        """

        if dataset == "MNIST":
            return self.__load_MNIST_dataset(DATASET_PERCENTAGE)
        elif dataset == "LFW":
            return self.__load_LFW_dataset()
        else:
            raise Exception("Bad `dataset` parameter given!")

    def __test_sampling_is_P_correct(self, dataset_selection: str = "MNIST"):
        """
        Test that the sampling we are doing respect the P-K philosophy relative to P.

        That's to say, check That every batch has exactly P classes
        """

        # Select the dataset we're going to use
        dataset = self.__get_dataset(dataset_selection)

        for P in range(1, 4):

            # Create dataloader with P classes. K is arbitrary
            # In this dataloader we specify our custom sampler
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size = P * 16,
                num_workers = NUM_WORKERS,
                pin_memory = True,
                sampler = CustomSampler(P, 16, dataset)
            )

            # Check the condition for every batch
            for _, label_batch in loader:

                # We dont care about not full batches
                if len(label_batch) < P * 16:
                    print("Skipping not full batch")
                    continue

                # Transform tensor of labels to list of int labels
                label_batch = [int(label) for label in label_batch]

                # Transform prev list to set to get unique labels
                unique_labels_in_batch = set(label_batch)

                # Assert we have exactly the number of expected labels
                self.assertEqual(len(unique_labels_in_batch), P, msg = "This batch doesn't contain exactly P classes")

    def test_sampling_is_P_correct(self):

        # Test for the two datasets
        self.__test_sampling_is_P_correct(dataset_selection = "MNIST")
        self.__test_sampling_is_P_correct(dataset_selection = "LFW")


    def __test_sampling_is_K_correct(self, dataset_selection: str = "MNIST"):
        """
        Test that the sampling we are doing respect the P-K philosophy relative to K.

        That's to say, check that every batch has exactly K elements for each class
        """

        # Dataset is going to be the same for all checks
        dataset = self.__get_dataset(dataset_selection)

        for K in range(1, 4):

            # Create dataloader with K images per classes. P is arbitrary
            # In this dataloader we specify our custom sampler
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size = 3 * K,
                num_workers = NUM_WORKERS,
                pin_memory = True,
                sampler = CustomSampler(3, K, dataset)
            )

            # Check the condition for every batch
            for _, label_batch in loader:

                # We dont care about not full batches
                if len(label_batch) < 3 * K:
                    print("Skipping not full batch")
                    continue

                # Transform tensor of labels to list of int labels
                label_batch = [int(label) for label in label_batch]

                # Transform prev list to set to get unique labels
                unique_labels_in_batch = set(label_batch)

                # Assert we have exactly the number of expected elements for each class
                # Use counter to not repeat computations for each class
                counter = Counter(label_batch)
                for label in unique_labels_in_batch:
                    self.assertEqual(counter[label], K, msg = "In this batch there is one class with not exactly K elements associated")

    def test_sampling_is_K_correct(self):

        # Test for the two datasets
        self.__test_sampling_is_K_correct(dataset_selection = "MNIST")
        self.__test_sampling_is_K_correct(dataset_selection = "LFW")

    def __test_list_of_classes_has_all_classes(self, dataset_selection: str = "MNIST"):
        """
        Check that the precomputation for list_of_classes generates a list of list with all 10 classes
        precomputed properly
        """

        # Create dataset and then directly a sampler (without dataloader)
        dataset = self.__get_dataset(dataset_selection)
        P, K = 3, 16
        sampler = CustomSampler(P, K, dataset)

        # Manually trigger the pre-computations
        # generate_index_sequence also pre-computes list_of_classes
        # TODO -- bad design because we have to know internal details
        sampler.generate_index_sequence()

        # Check that list of classes has all classes pre-computed
        self.assertEqual(len(sampler.dict_of_classes), 10)

    def test_list_of_classes_has_all_classes(self):

        # Test for the two datasets
        self.__test_list_of_classes_has_all_classes(dataset_selection = "MNIST")
        self.__test_list_of_classes_has_all_classes(dataset_selection = "LFW")

    # TODO -- what was 'the cleaning mechanism'??
    def __test_remove_empty_classes(self, dataset_selection: str = "MNIST"):
        """Check that the cleaning mechanism for available classes works fine"""

        # Create dataset and then directly a sampler (without dataloader)
        dataset = self.__get_dataset(dataset_selection)
        P, K = 3, 16
        sampler = CustomSampler(P, K, dataset)

        # Now manually create a dict of classes
        # This classes should survive cleanning
        sampler.dict_of_classes = dict()
        sampler.dict_of_classes[0] = list(range(80))
        sampler.dict_of_classes[1] = list(range(80))
        sampler.dict_of_classes[2] = list(range(80))
        sampler.dict_of_classes[5] = list(range(80))
        sampler.dict_of_classes[6] = list(range(80))
        sampler.dict_of_classes[7] = list(range(80))
        sampler.dict_of_classes[8] = list(range(80))

        # This classes should not survive cleaning
        sampler.dict_of_classes[3] = [1, 2, 3]
        sampler.dict_of_classes[4] = [1]
        sampler.dict_of_classes[9] = [1, 2, 3, 4]

        # Clean and check the list
        cleaned_list_of_classes = sampler.remove_empty_classes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(len(cleaned_list_of_classes), 10 - 3)
        self.assertIn(0, cleaned_list_of_classes)
        self.assertIn(1, cleaned_list_of_classes)
        self.assertIn(2, cleaned_list_of_classes)
        self.assertNotIn(3, cleaned_list_of_classes)
        self.assertNotIn(4, cleaned_list_of_classes)
        self.assertIn(5, cleaned_list_of_classes)
        self.assertIn(6, cleaned_list_of_classes)
        self.assertIn(7, cleaned_list_of_classes)
        self.assertIn(8, cleaned_list_of_classes)
        self.assertNotIn(9, cleaned_list_of_classes)

        # Repeat the process with other list of classes
        # This classes should survive cleanning
        sampler.dict_of_classes[0] = list(range(80))
        sampler.dict_of_classes[3] = list(range(80))
        sampler.dict_of_classes[4] = list(range(80))
        sampler.dict_of_classes[5] = list(range(80))
        sampler.dict_of_classes[6] = list(range(80))
        sampler.dict_of_classes[7] = list(range(80))
        sampler.dict_of_classes[8] = list(range(80))
        sampler.dict_of_classes[9] = list(range(80))

        # This classes should not survive cleaning
        sampler.dict_of_classes[1] = [1, 2, 3]
        sampler.dict_of_classes[2] = [1]

        # Clean and check the list
        cleaned_list_of_classes = sampler.remove_empty_classes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(len(cleaned_list_of_classes), 10 - 2)
        self.assertIn(0, cleaned_list_of_classes)
        self.assertNotIn(1, cleaned_list_of_classes)
        self.assertNotIn(2, cleaned_list_of_classes)
        self.assertIn(3, cleaned_list_of_classes)
        self.assertIn(4, cleaned_list_of_classes)
        self.assertIn(5, cleaned_list_of_classes)
        self.assertIn(6, cleaned_list_of_classes)
        self.assertIn(7, cleaned_list_of_classes)
        self.assertIn(8, cleaned_list_of_classes)
        self.assertIn(9, cleaned_list_of_classes)

    def test_remove_empty_classes(self):

        # Test for the two datasets
        self.__test_remove_empty_classes(dataset_selection = "MNIST")
        self.__test_remove_empty_classes(dataset_selection = "LFW")


    def __test_len_computation_is_correct(self, dataset_selection: str = "MNIST"):
        """
        Due to P-K sampling, not all elements of the dataset are sampled

        With this test, I am looking that CustomSampler.__len__ computes how many elements are
        sampled properly

        """

        # Create a dataset
        dataset = self.__get_dataset(dataset_selection)

        # Create a sampler with certain values of P, K
        P, K = 3, 32 if dataset_selection == "MNIST" else 2
        sampler = CustomSampler(P, K, dataset)

        # Check that CustomSampler.__len__ computation is correct
        sampled_elements = [element for element in sampler]
        self.assertEqual(
            len(sampled_elements),
            len(sampler),
            msg = "Len of returned elements of the sampler is not equal to CustomSampler.__len__ computation"
        )

        # Repeat for other values of P, K
        P, K = 3, 16 if dataset_selection == "MNIST" else 3
        sampler = CustomSampler(P, K, dataset)

        sampled_elements = [element for element in sampler]
        self.assertEqual(
            len(sampled_elements),
            len(sampler),
            msg = "Len of returned elements of the sampler is not equal to CustomSampler.__len__ computation"
        )

    def test_len_computation_is_correct(self):

        # Test for the two datasets
        self.__test_len_computation_is_correct(dataset_selection = "MNIST")
        self.__test_len_computation_is_correct(dataset_selection = "LFW")


    def __test_sampler_len_is_less_than_dataset_len(self, dataset_selection: str = "MNIST"):
        """
        Check that sampler len is less or equal than dataset len

        As we explain in CustomSampler.__len__ docs, this inequality has to hold because some
        elements of the dataset can be not sampled
        """

        # Create a dataset
        dataset = self.__get_dataset(dataset_selection)

        # Create a sampler with certain values of P, K
        P, K = 3, 32
        sampler = CustomSampler(P, K, dataset)

        # Check that CustomSampler.__len__ computation is correct
        # We need to sample elements to trigger len computation
        sampled_elements = [element for element in sampler]
        self.assertLessEqual(
            len(sampler),
            len(dataset),
            msg = "Len of sampler has to be less or equal to len of dataset"
        )

        # Repeat for other values of P, K
        P, K = 3, 16
        sampler = CustomSampler(P, K, dataset)

        # We need to sample elements to trigger len computation
        sampled_elements = [element for element in sampler]
        self.assertLessEqual(
            len(sampler),
            len(dataset),
            msg = "Len of sampler has to be less or equal to len of dataset"
        )

    def test_sampler_len_is_less_than_dataset_len(self):

        # Test for the two datasets
        self.__test_sampler_len_is_less_than_dataset_len(dataset_selection = "MNIST")
        self.__test_sampler_len_is_less_than_dataset_len(dataset_selection = "LFW")
