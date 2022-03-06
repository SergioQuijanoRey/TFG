import unittest

import torch
import torchvision
import torchvision.transforms as transforms

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

        for P in range(1, 10):

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

        for K in range(1, 10):

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
                for label in unique_labels_in_batch:
                    self.assertEqual(label_batch.count(label), K)

