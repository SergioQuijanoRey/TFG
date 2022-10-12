import unittest
import torch

import src.lib.metrics as metrics
import src.lib.utils as utils

class TestComputeIntraclusterDistances(unittest.TestCase):

    def test_basic_case(self):
        targets = [0, 0, 0, 1, 1, 1]
        images = torch.Tensor([
            # Class 0 images
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],

            # Class 1 images
            [1, 0, 0],
            [3, 1, 0],
            [10, 0, 0],
        ])

        # Compute the intracluster distances
        dict_of_classes = utils.precompute_dict_of_classes(targets)
        intra_cluster_distances = metrics.compute_intracluster_distances(dict_of_classes, images)

        # Distances corresponding to the first cluster
        self.assertAlmostEqual(intra_cluster_distances[0][0], 1.0)
        self.assertAlmostEqual(intra_cluster_distances[0][1], 2.0)
        self.assertAlmostEqual(intra_cluster_distances[0][2], 1.0)

        # Distances corresponding to the second cluster
        self.assertAlmostEqual(intra_cluster_distances[1][0], 2.23606797749979)
        self.assertAlmostEqual(intra_cluster_distances[1][1], 9.0)
        self.assertAlmostEqual(intra_cluster_distances[1][2], 7.0710678118654755)
