"""
Testing of loss functions
"""

import unittest
import torch
import math

from src.lib.loss_functions import distance_function

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
