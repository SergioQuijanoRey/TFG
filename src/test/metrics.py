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

    def test_clusters_with_one_element_produce_no_distance(self):
        targets = [0, 1, 2, 3, 4, 5]
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
        self.assertEqual(len(intra_cluster_distances), 0)


    def test_clusters_with_two_elements_produce_distance(self):
        targets = [0, 0, 1, 1, 2, 2]
        images = torch.Tensor([
            # Class 0 images
            [0, 0, 0],
            [0, 0, 1],

            # Class 1 images
            [0, 0, 2],
            [1, 0, 0],

            # Class 2 images
            [3, 1, 0],
            [10, 0, 0],
        ])

        # Compute the intracluster distances
        dict_of_classes = utils.precompute_dict_of_classes(targets)
        intra_cluster_distances = metrics.compute_intracluster_distances(dict_of_classes, images)

        # Some checks
        self.assertEqual(len(intra_cluster_distances), 3)
        self.assertAlmostEqual(intra_cluster_distances[0][0], 1.0)
        self.assertAlmostEqual(intra_cluster_distances[1][0], 2.23606797749979)
        self.assertAlmostEqual(intra_cluster_distances[2][0], 7.0710678118654755)

class TestComputeClusterSizesMetrics(unittest.TestCase):

    def __generate_basic_dataset(self) -> torch.utils.data.Dataset:
        targets = torch.Tensor([0, 0, 0, 1, 1, 1])
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

        dataset = torch.utils.data.TensorDataset(images, targets)
        return dataset



    def test_basic_case(self):

        dataset = self.__generate_basic_dataset()
        dataloader = torch.utils.data.DataLoader(dataset)
        net = torch.nn.Identity()

        # cluter_sizes =  2.0 9.0
        cluster_metrics = metrics.compute_cluster_sizes_metrics(dataloader, net, 6)

        # Make some checks
        self.assertAlmostEqual(cluster_metrics["min"], 2.0)
        self.assertAlmostEqual(cluster_metrics["max"], 9.0)
        self.assertAlmostEqual(cluster_metrics["mean"], 5.5)
        self.assertAlmostEqual(cluster_metrics["sd"], 3.5)

