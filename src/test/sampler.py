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

class TestCustomSampler(unittest.TestCase):

    def __load_dataset(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.MNIST(
            root = DATA_PATH,
            train = True,
            download = True,
            transform = transform,
        )

        return dataset


    def test_sampling_is_P_correct(self):
        """
        Test that the sampling we are doing respect the P-K philosophy relative to P.

        That's to say, check That every batch has exactly P classes
        """
        self.assertEqual(1, 2-1)

        # Dataset is going to be the same for all checks
        dataset = self.__load_dataset()

        for P in range(1, 4):

            # Create dataloader with P classes. K is arbitrary
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
                self.assertEqual(len(unique_labels_in_batch), P)


    def test_sampling_is_K_correct(self):
        """
        Test that the sampling we are doing respect the P-K philosophy relative to K.

        That's to say, check that every batch has exactly K elements for each class
        """
        self.assertEqual(1, 2-1)

        # Dataset is going to be the same for all checks
        dataset = self.__load_dataset()

        for K in range(1, 4):

            # Create dataloader with K images per classes. P is arbitrary
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
                    self.assertEqual(counter[label], K)

    def test_list_of_classes_has_all_classes(self):
        """
        Check that the precomputation for list_of_classes generates a list of list with all 10 classes
        precomputed properly
        """
        # Create dataset and then directly a sampler (without dataloader)
        dataset = self.__load_dataset()
        P, K = 3, 16
        sampler = CustomSampler(P, K, dataset)

        # Manually trigger the pre-computations
        # generate_index_sequence also pre-computes list_of_classes
        # TODO -- bad design because we have to know internal details
        sampler.generate_index_sequence()

        # Check that list of classes has all classes pre-computed
        self.assertEqual(len(sampler.list_of_classes), 10)

    def test_clean_available_classes(self):
        """Check that the cleaning mechanism for available classes works fine"""

        # Create dataset and then directly a sampler (without dataloader)
        dataset = self.__load_dataset()
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
        cleaned_list_of_classes = sampler.clean_list_of_classes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
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
        cleaned_list_of_classes = sampler.clean_list_of_classes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
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
