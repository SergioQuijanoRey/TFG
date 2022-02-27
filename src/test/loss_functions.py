"""
Testing of loss functions
"""

import unittest
import torch
import math

from src.lib.loss_functions import distance_function, TripletLoss, SoftplusTripletLoss, BatchHardTripletLoss

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


class TestSoftplusTripletLoss(unittest.TestCase):

    def test_basic_cases(self):
        softplus_loss = SoftplusTripletLoss()

        # First basic example
        anchor = torch.tensor([0.0, 0.0])
        positive = torch.tensor([1.0, 1.0])
        negative = torch.tensor([2.0, 2.0])

        loss_computed = float(softplus_loss(anchor, positive, negative))
        loss_expected = 0.21762172158174375
        self.assertAlmostEqual(loss_computed, loss_expected)

        # Second basic example
        anchor = torch.tensor([0.0, 0.0])
        positive = torch.tensor([2.0, 2.0])
        negative = torch.tensor([2.0, 2.0])

        loss_computed = float(softplus_loss(anchor, positive, negative))
        loss_expected = 0.6931471805599453
        self.assertAlmostEqual(loss_computed, loss_expected)

        # Third basic example
        anchor = torch.tensor([0.0, 0.0])
        positive = torch.tensor([1.0, 1.0])
        negative = torch.tensor([-4.0, 2.0])

        loss_computed = float(softplus_loss(anchor, positive, negative))
        loss_expected = 0.04591480638813308
        self.assertAlmostEqual(loss_computed, loss_expected)

        # Fourth basic example
        anchor = torch.tensor([0.0, 0.0])
        positive = torch.tensor([-4.0, 2.0])
        negative = torch.tensor([1.0, 1.0])

        loss_computed = float(softplus_loss(anchor, positive, negative))
        loss_expected = 3.1038371990146176
        self.assertAlmostEqual(loss_computed, loss_expected, places = 1)

class TestBatchHardTripletLoss(unittest.TestCase):

    def test_basic_cases(self):

        # Define the loss function to test
        loss = BatchHardTripletLoss(
            margin = 1.0,
            use_softplus = False,
            use_gt_than_zero_mean = False
        )

        # Define some fixed data to test on
        points = torch.tensor([
            # 0-th class points
            [0, 0], [2, 2], [9, -6],

            # 1-th class points
            [7, -4], [12, -4], [2, -9],

            # 2-th class points
            [1, -11], [4, -11], [-1, 2]
        ])

        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

        loss_computed = float(loss(points, labels))
        loss_expected = 9.741091053890488
        self.assertAlmostEqual(loss_computed, loss_expected)

    def test_basic_cases_gt_than_zero(self):

        # Define the loss function to test
        loss = BatchHardTripletLoss(
            margin = 1.0,
            use_softplus = False,
            use_gt_than_zero_mean = True
        )

        # Define some fixed data to test on
        points = torch.tensor([
            # 0-th class points
            [0, 0], [2, 2], [9, -6],

            # 1-th class points
            [7, -4], [12, -4], [2, -9],

            # 2-th class points
            [1, -11], [4, -11], [-1, 2]
        ])

        labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])

        loss_computed = float(loss(points, labels))
        loss_expected = 9.741091053890488
        self.assertAlmostEqual(loss_computed, loss_expected)
