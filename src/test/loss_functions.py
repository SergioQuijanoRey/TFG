"""
Testing of loss functions
"""

import unittest
import torch
import math

from src.lib.loss_functions import distance_function, TripletLoss

# Number of repetitions, when needed
NUMBER_OF_REPETITIONS = 100

class TestBasicLossFunction(unittest.TestCase):
    """Tests about basic distance function"""

    def test_distances_greater_than_zero(self):

        for _ in range(NUMBER_OF_REPETITIONS):

            # Generate two random tensors
            first = torch.rand(2)
            second = torch.rand(2)

            # Compute distance
            distance = distance_function(first, second)

            # Assert distance is greater than zero
            self.assertGreater(distance, 0.0)

        for _ in range(NUMBER_OF_REPETITIONS):

            # Generate two random tensors
            first = torch.rand(4)
            second = torch.rand(4)

            # Compute distance
            distance = distance_function(first, second)

            # Assert distance is greater than zero
            self.assertGreater(distance, 0.0)

    def test_some_known_examples(self):

        # First basic example
        first = torch.tensor([0.0, 0.0])
        second = torch.tensor([1.0, 1.0])

        dist_computed = distance_function(first, second)
        dist_expect = math.sqrt(2.0)
        self.assertAlmostEquals(dist_computed, dist_expect)

        # Second basic example
        first = torch.tensor([1.0, 1.0])
        second = torch.tensor([2.0, 2.0])

        dist_computed = distance_function(first, second)
        dist_expect = math.sqrt(2.0)
        self.assertAlmostEquals(dist_computed, dist_expect)

        # Third basic example
        first = torch.tensor([1.0, 1.0])
        second = torch.tensor([-4.0, 2.0])

        dist_computed = distance_function(first, second)
        dist_expect = math.sqrt(26)
        self.assertAlmostEquals(dist_computed, dist_expect)

class TestTripletLoss(unittest.TestCase):

    def test_basic_cases(self):
        margin = 1.0
        triplet_loss = TripletLoss(margin = margin)

        # First basic example
        anchor = torch.tensor([0.0, 0.0])
        positive = torch.tensor([1.0, 1.0])
        negative = torch.tensor([2.0, 2.0])

        loss_computed = float(triplet_loss(anchor, positive, negative))
        loss_expected = margin + math.sqrt(2) - math.sqrt(8)
        loss_expected = loss_expected if loss_expected >= 0 else 0
        self.assertAlmostEqual(loss_computed, loss_expected)

        # Second basic example
        anchor = torch.tensor([0.0, 0.0])
        positive = torch.tensor([2.0, 2.0])
        negative = torch.tensor([2.0, 2.0])

        loss_computed = float(triplet_loss(anchor, positive, negative))
        loss_expected = margin
        loss_expected = loss_expected if loss_expected >= 0 else 0
        self.assertAlmostEqual(loss_computed, loss_expected)

        # Third basic example
        anchor = torch.tensor([0.0, 0.0])
        positive = torch.tensor([1.0, 1.0])
        negative = torch.tensor([-4.0, 2.0])

        loss_computed = float(triplet_loss(anchor, positive, negative))
        loss_expected = 0.0
        loss_expected = loss_expected if loss_expected >= 0 else 0
        self.assertAlmostEqual(loss_computed, loss_expected)

        # Fourth basic example
        anchor = torch.tensor([0.0, 0.0])
        positive = torch.tensor([-4.0, 2.0])
        negative = torch.tensor([1.0, 1.0])

        loss_computed = float(triplet_loss(anchor, positive, negative))
        loss_expected = margin + math.sqrt(20) - math.sqrt(2)
        loss_expected = loss_expected if loss_expected >= 0 else 0
        self.assertAlmostEqual(loss_computed, loss_expected)
