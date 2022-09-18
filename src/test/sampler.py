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

    def __load_dataset(self, percentage: float = 1.0):
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

    def test_sampling_is_P_correct(self):
        """
        Test that the sampling we are doing respect the P-K philosophy relative to P.

        That's to say, check That every batch has exactly P classes
        """

        # Dataset is going to be the same for all checks
        dataset = self.__load_dataset(DATASET_PERCENTAGE)

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


    def test_sampling_is_K_correct(self):
        """
        Test that the sampling we are doing respect the P-K philosophy relative to K.

        That's to say, check that every batch has exactly K elements for each class
        """

        # Dataset is going to be the same for all checks
        dataset = self.__load_dataset(DATASET_PERCENTAGE)

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

    def test_list_of_classes_has_all_classes(self):
        """
        Check that the precomputation for list_of_classes generates a list of list with all 10 classes
        precomputed properly
        """

        # Create dataset and then directly a sampler (without dataloader)
        dataset = self.__load_dataset(DATASET_PERCENTAGE)
        P, K = 3, 16
        sampler = CustomSampler(P, K, dataset)

        # Manually trigger the pre-computations
        # generate_index_sequence also pre-computes list_of_classes
        # TODO -- bad design because we have to know internal details
        sampler.generate_index_sequence()

        # Check that list of classes has all classes pre-computed
        self.assertEqual(len(sampler.list_of_classes), 10)

    # TODO -- what was 'the cleaning mechanism'??
    def test_remove_empty_classes(self):
        """Check that the cleaning mechanism for available classes works fine"""

        # Create dataset and then directly a sampler (without dataloader)
        dataset = self.__load_dataset(DATASET_PERCENTAGE)
        P, K = 3, 16
        sampler = CustomSampler(P, K, dataset)

        # Now manually create list of classes
        # This classes should survive cleanning
        sampler.list_of_classes = [[] for _ in range(10)]
        sampler.list_of_classes[0] = list(range(80))
        sampler.list_of_classes[1] = list(range(80))
        sampler.list_of_classes[2] = list(range(80))
        sampler.list_of_classes[5] = list(range(80))
        sampler.list_of_classes[6] = list(range(80))
        sampler.list_of_classes[7] = list(range(80))
        sampler.list_of_classes[8] = list(range(80))

        # This classes should not survive cleaning
        sampler.list_of_classes[3] = [1, 2, 3]
        sampler.list_of_classes[4] = [1]
        sampler.list_of_classes[9] = [1, 2, 3, 4]

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
        sampler.list_of_classes[0] = list(range(80))
        sampler.list_of_classes[3] = list(range(80))
        sampler.list_of_classes[4] = list(range(80))
        sampler.list_of_classes[5] = list(range(80))
        sampler.list_of_classes[6] = list(range(80))
        sampler.list_of_classes[7] = list(range(80))
        sampler.list_of_classes[8] = list(range(80))
        sampler.list_of_classes[9] = list(range(80))

        # This classes should not survive cleaning
        sampler.list_of_classes[1] = [1, 2, 3]
        sampler.list_of_classes[2] = [1]

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


    # TODO -- BUG -- this test is failing
    # TODO -- I think expecting all elements to be returned always is incorrect
    #         If len(dataset) % (P * K) != 0 then this can fail
    #         If the sampler returns a last batch with all elements remaining, then prev tests will
    #         fail (like P, K tests)
    def test_all_elements_are_returned(self):
        """
        Check that the sampler returns all elements of the dataset

        With this test, I am looking if the sampler lefts behind some elements when P, K doesn't
        fit well the dataset size
        """

        # Create a dataset
        dataset = self.__load_dataset(DATASET_PERCENTAGE)

        # Create a sampler with certain values of P, K
        P, K = 3, 16
        sampler = CustomSampler(P, K, dataset)

        # Check that all elements are in the dataset
        sampled_elements = [element for element in sampler]
        self.assertEqual(len(sampled_elements), len(dataset))

        # Repeat for other values of P, K
        P, K = 5, 32
        sampler = CustomSampler(P, K, dataset)

        sampled_elements = [element for element in sampler]
        self.assertEqual(
            len(sampled_elements),
            len(dataset),
            msg = "Some elements of the dataset were not returned when using the custom sampler"
        )
