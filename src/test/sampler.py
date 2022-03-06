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


    def test_sampling_is_p_correct(self):
        """
        Test that the sampling we are doing respect the P-K philosophy relative to P. That's to say,
        check That every batch has exactly P classes
        """
        self.assertEqual(1, 2-1)

        # Dataset is going to be the same for all checks
        dataset = self.__load_dataset()

        for P in range(1, 10):

            # Create dataloader with P classes. K is arbitrary
            train_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size = BATCH_SIZE,
                #shuffle = True,
                num_workers = NUM_WORKERS,
                pin_memory = True,
                # TODO -- using magic numbers
                sampler = CustomSampler(P, 16, dataset)
            )

            for img_batch, label_batch in train_loader:

                # Transform tensor of labels to list of int labels
                label_batch = [int(label) for label in label_batch]

                # Transform prev list to set to get unique labels
                unique_labels_in_batch = set(label_batch)

                # Assert we have exactly the number of expected labels
                self.assertEqual(len(unique_labels_in_batch), P)

