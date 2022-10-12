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

        # Compute the cluster metrics
        cluster_metrics = metrics.compute_cluster_sizes_metrics(dataloader, net, 6)

        # Make some checks
        self.assertAlmostEqual(cluster_metrics["min"], 2.0)
        self.assertAlmostEqual(cluster_metrics["max"], 9.0)
        self.assertAlmostEqual(cluster_metrics["mean"], 5.5)
        self.assertAlmostEqual(cluster_metrics["sd"], 3.5)

class TestComputeInterclusterDistances(unittest.TestCase):

    def test_basic_case(self):

        # Basic data for the test
        dict_of_classes = {
            0: [0, 1],
            1: [2, 3],
            2: [4, 5]
        }
        distances = {
            (0, 1): 1,
            (0, 2): 2,
            (0, 3): 3,
            (0, 4): 1.5,
            (0, 5): 3.4,

            (1, 2): 6,
            (1, 3): 1.1,
            (1, 4): 0.5,
            (1, 5): 1.1111,

            (2, 3): 2.222,
            (2, 4): 3.333,
            (2, 5): 4.444,

            (3, 4): 5.555,
            (3, 5): 0.102,

            (4, 5): 0.01,
        }

        # Compute the intercluster distances
        intercluster_distances = metrics.compute_intercluster_distances(distances, dict_of_classes)

        # Make some checks on the returned data
        self.assertAlmostEqual(intercluster_distances[0, 1], 1.1)
        self.assertAlmostEqual(intercluster_distances[0, 2], 0.5)
        self.assertAlmostEqual(intercluster_distances[1, 2], 0.102)

class TestComputeInterclusterMetrics(unittest.TestCase):

    def __generate_basic_dataset(self) -> torch.utils.data.Dataset:
        targets = torch.Tensor([0, 0, 1, 1, 2, 2])
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

        return torch.utils.data.TensorDataset(images, targets)


    def test_basic_case(self):

        # Get the data into a dataloader
        dataset = self.__generate_basic_dataset()
        dataloader = torch.utils.data.DataLoader(dataset)

        # Identity net for not mutating the mock data
        net = torch.nn.Identity()

        # Get the metrics
        intercluster_metrics = metrics.compute_intercluster_metrics(dataloader, net, 6)

        # Make some checks on the metrics
        self.assertAlmostEqual(intercluster_metrics["min"], 1.0, places = 4)
        self.assertAlmostEqual(intercluster_metrics["max"], 3.1623, places = 4)
        self.assertAlmostEqual(intercluster_metrics["mean"], 2.1328, places = 4)

