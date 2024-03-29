import unittest
import torch
import torchvision
import os
import torchvision.transforms as transforms

import src.lib.split_dataset as split_dataset
import src.lib.datasets as datasets

class TestSplitDataset(unittest.TestCase):

    def test_lfw_dataset(self):

        # Load the dataset
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((250, 250), antialias=True),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])
        dataset = torchvision.datasets.LFWPeople(
            root = "./data",
            split = "train",
            download = True,
            transform = transform,
        )

        # Split the dataset
        first, second = split_dataset.split_dataset(dataset, 0.5)

        # Check that subsets has the attribute target
        self.assertTrue(hasattr(first, "targets"), "Splitting does not preserve `targets` attribute")
        self.assertTrue(hasattr(second, "targets"), "Splitting does not preserve `targets` attribute")

        # Now, check that targets are half the size
        self.assertAlmostEqual(len(first.targets), len(dataset.targets) / 2, 0, "Split does not treat sizes properly")
        self.assertAlmostEqual(len(second.targets), len(dataset.targets) / 2, 0, "Split does not treat sizes properly")

        # Now, check that the whole dataset len is correct
        self.assertAlmostEqual(len(first), len(dataset) / 2, 0, "Split does not treat sizes properly")
        self.assertAlmostEqual(len(second), len(dataset) / 2, 0, "Split does not treat sizes properly")

class TestSplitDatasetDisjoint(unittest.TestCase):

    def __get_fg_dataset(self) -> torch.utils.data.Dataset:
        DATA_PATH = "./data"
        DATASET_URL = "http://yanweifu.github.io/FG_NET_data/FGNET.zip"

        datasets.download_fg_dataset(
            DATA_PATH,
            DATASET_URL,
            can_skip_download = True
        )

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels = 3),
            transforms.Resize((300, 300), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
             ),
        ])

        dataset = datasets.FGDataset(path = os.path.join(DATA_PATH, "FGNET/images"), transform = transform)
        return dataset


    def test_fg_net_sizes(self):

        # Get a dataset and split it
        dataset = self.__get_fg_dataset()
        first_dataset, second_dataset = split_dataset.split_dataset_disjoint_classes(
            dataset, 0.7
        )

        # The percentage is not going to be accurate, but it should respect a bit
        first_perc = len(first_dataset) / len(dataset)
        second_perc = len(second_dataset) / len(dataset)

        # We are only failing at most at plus/minus 5%
        epsilon = 0.1

        self.assertAlmostEqual(
            first_perc,
            0.7,
            places = None,
            msg = "First dataset is way too small",
            delta = epsilon
        )
        self.assertAlmostEqual(
            second_perc,
            0.3,
            places = None,
            msg = "Second dataset is way too big",
            delta = epsilon
        )

    def test_fg_net_disjoint(self):

        # Get a dataset and split it
        dataset = self.__get_fg_dataset()
        first_dataset, second_dataset = split_dataset.split_dataset_disjoint_classes(
            dataset, 0.7
        )

        first_targets = first_dataset.targets
        second_targets = second_dataset.targets

        # One pass should be enough. That's to say, we dont iterate over
        # targets of the second dataset, and check against the first dataset
        for target in first_targets:

            if target in second_targets:
                msg = f"Target {target} found in both datasets!"
                raise Exception(msg)

    def test_no_element_is_lost(self):
        # Get a dataset and split it
        dataset = self.__get_fg_dataset()
        first_dataset, second_dataset = split_dataset.split_dataset_disjoint_classes(
            dataset, 0.7
        )

        # Then split the second dataset again
        second_dataset, third_dataset = split_dataset.split_dataset_disjoint_classes(
            second_dataset, 0.5
        )

        # Compute the sum of the elements of three datasets
        split_total_len = len(first_dataset) + len(second_dataset) + len(third_dataset)
        self.assertEqual(split_total_len, len(dataset), "Some elements got lost or got repeated!")

    def test_percentages_sum_up_to_one(self):

        # Get a dataset and split it
        dataset = self.__get_fg_dataset()
        first_dataset, second_dataset = split_dataset.split_dataset_disjoint_classes(
            dataset, 0.7
        )

        # Then split the second dataset again
        second_dataset, third_dataset = split_dataset.split_dataset_disjoint_classes(
            second_dataset, 0.5
        )

        # Compute the percentages
        first_perc = len(first_dataset) / len(dataset)
        second_perc = len(second_dataset) / len(dataset)
        third_perc = len(third_dataset) / len(dataset)

        # Check that percentages sum up to one
        sum_perc = first_perc + second_perc + third_perc
        self.assertAlmostEqual(
            sum_perc,
            1.0,
            places = None,
            msg = f"Percentages don't sum up to one!\n{first_perc=} :: {second_perc=} :: {third_perc=}",
            delta = 0.01,
        )
