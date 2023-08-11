import unittest
import torch
import torchvision

from src.lib.split_dataset import split_dataset, WrappedSubset

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
        first, second = split_dataset(dataset, 0.5)

        # Check that subsets has the attribute target
        self.assertTrue(hasattr(first, "targets"), "Splitting does not preserve `targets` attribute")
        self.assertTrue(hasattr(second, "targets"), "Splitting does not preserve `targets` attribute")

        # Now, check that targets are half the size
        self.assertAlmostEqual(len(first.targets), len(dataset.targets) / 2, 0, "Split does not treat sizes properly")
        self.assertAlmostEqual(len(second.targets), len(dataset.targets) / 2, 0, "Split does not treat sizes properly")

        # Now, check that the whole dataset len is correct
        self.assertAlmostEqual(len(first), len(dataset) / 2, 0, "Split does not treat sizes properly")
        self.assertAlmostEqual(len(second), len(dataset) / 2, 0, "Split does not treat sizes properly")
