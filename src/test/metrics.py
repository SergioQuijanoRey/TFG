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

    def test_single_element_clusters(self):
        # Basic data for the test
        dict_of_classes = {
            0: [0],
            1: [1],
            2: [2],
            3: [3],
            4: [4],
            5: [5],
        }

        # This distances correspond to the `TestComputeInterclusterMetrics.__generate_basic_dataset`
        # but having each image its own cluster (single element clusters)
        # I have a problem in `TestComputeInterclusterMetrics.test_single_element_clusters`, so
        # this test is helping me debug this problem
        distances = {
            (0, 1): 0,
            (0, 2): 2,
            (0, 3): 1,
            (0, 4): 3.1622776601683795,
            (0, 5): 10,

            (1, 2): 1.0,
            (1, 3): 1.4142135623730951,
            (1, 4): 3.3166247903554,
            (1, 5): 10.04987562112089,

            (2, 3): 2.23606797749979,
            (2, 4): 3.7416573867739413,
            (2, 5): 10.198039027185569,

            (3, 4): 2.23606797749979,
            (3, 5): 9.0,

            (4, 5): 7.0710678118654755,
        }

        # Compute the intercluster distances
        intercluster_distances = metrics.compute_intercluster_distances(distances, dict_of_classes)

        # Check that the i-j cluster distance is the distance between the single elements of each
        # one of that cluster, that's to say, between points i-j
        for first_cluster in range(5):
            for second_cluster in range(5):

                if first_cluster >= second_cluster:
                    continue

                self.assertAlmostEqual(
                    intercluster_distances[first_cluster, second_cluster],
                    distances[first_cluster, second_cluster],
                    "Cluster distance should be the same as point distance, for single-element clusters"
                )

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

    def test_single_element_clusters(self):
        # Get the basic dataset
        dataset = self.__generate_basic_dataset()

        # Change the labels of the dataset
        # Each element is in its own cluster
        dataset.targets = torch.Tensor([0, 1, 2, 3, 4, 5, 6])

        # Wrap dataset into a dataloader
        dataloader = torch.utils.data.DataLoader(dataset)

        # Identity network
        net = torch.nn.Identity()

        # Get the metrics
        intercluster_metrics = metrics.compute_intercluster_metrics(dataloader, net, 6)

        # Make some checks about the obtained metrics

        self.assertAlmostEqual(intercluster_metrics["mean"], 4.495059454322822)
        self.assertAlmostEqual(intercluster_metrics["min"], 1.0)
        self.assertAlmostEqual(intercluster_metrics["max"], 10.198039027185569)
        self.assertAlmostEqual(intercluster_metrics["sd"], 3.653927510536194)
