import unittest
import torch
import torchvision
import torchvision.transforms as transforms

import src.lib.metrics as metrics
import src.lib.utils as utils
import src.lib.data_augmentation as data_augmentation
import src.lib.sampler as sampler
import src.lib.models as models

# Precision that we want in certain tests
PLACES = 4

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
        self.assertAlmostEqual(intra_cluster_distances[0][0], 1.0, places = PLACES)
        self.assertAlmostEqual(intra_cluster_distances[0][1], 2.0, places = PLACES)
        self.assertAlmostEqual(intra_cluster_distances[0][2], 1.0, places = PLACES)

        # Distances corresponding to the second cluster
        self.assertAlmostEqual(intra_cluster_distances[1][0], 2.23606797749979, places = PLACES)
        self.assertAlmostEqual(intra_cluster_distances[1][1], 9.0, places = PLACES)
        self.assertAlmostEqual(intra_cluster_distances[1][2], 7.0710678118654755, places = PLACES)

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
        self.assertAlmostEqual(
            intra_cluster_distances[0][0],
            1.0,
            places = PLACES
        )
        self.assertAlmostEqual(
            intra_cluster_distances[1][0],
            2.23606797749979,
            places = PLACES
        )
        self.assertAlmostEqual(
            intra_cluster_distances[2][0],
            7.0710678118654755,
            places = PLACES
        )

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
        self.assertAlmostEqual(cluster_metrics["min"], 2.0, places = PLACES)
        self.assertAlmostEqual(cluster_metrics["max"], 9.0, places = PLACES)
        self.assertAlmostEqual(cluster_metrics["mean"], 5.5, places = PLACES)
        self.assertAlmostEqual(cluster_metrics["sd"], 3.5, places = PLACES)

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
        self.assertAlmostEqual(intercluster_distances[0, 1], 1.1, places = PLACES)
        self.assertAlmostEqual(intercluster_distances[0, 2], 0.5, places = PLACES)
        self.assertAlmostEqual(intercluster_distances[1, 2], 0.102, places = PLACES)

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

# Portion of the LFW dataset that we're going to use to test metrics.compute_intercluster_metrics()
DATASET_PORTION = 0.001

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
        self.assertAlmostEqual(intercluster_metrics["min"], 1.0, places = PLACES)
        self.assertAlmostEqual(intercluster_metrics["max"], 3.1623, places = PLACES)
        self.assertAlmostEqual(intercluster_metrics["mean"], 2.1328, places = PLACES)

    def test_single_element_clusters(self):

        # Get the same data as in __generate_basic_dataset, but changing the labels
        # We cannot get the dataset and change `dataset.targets` because in TensorDataset
        # this doesn't work
        targets = torch.Tensor([0, 1, 2, 3, 4, 5])
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

        # Wrap dataset into a dataloader
        dataloader = torch.utils.data.DataLoader(dataset)

        # Identity network
        net = torch.nn.Identity()

        # Get the metrics
        intercluster_metrics = metrics.compute_intercluster_metrics(dataloader, net, 6)

        # Make some checks about the obtained metrics
        self.assertAlmostEqual(intercluster_metrics["mean"], 4.495059454322822, places = PLACES)
        self.assertAlmostEqual(intercluster_metrics["min"], 1.0, places = PLACES)
        self.assertAlmostEqual(intercluster_metrics["max"], 10.198039027185569, places = PLACES)
        self.assertAlmostEqual(intercluster_metrics["sd"], 3.5300293438964045, places = PLACES)

    def test_lfw_dataset_works_basic(self):
        """
        I had problems running `compute_intercluster_metrics` for a portion of the LFW dataset. So
        in this test we're simply try to compute the values without any crash

        We're using a random net to test the behaviour
        """

        # Load the dataset
        transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(
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

        # Apply data augmentation for having at least 4 images per class
        augmented_dataset = data_augmentation.LazyAugmentatedDataset(
            base_dataset = dataset,
            min_number_of_images = 4,

            # Remember that the trasformation has to be random type
            # Otherwise, we could end with a lot of repeated images
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(250, 250)),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAutocontrast(),
            ])

        )

        # Now put a loader in front of the augmented dataset
        dataloader = torch.utils.data.DataLoader(
            augmented_dataset,
            batch_size = 3 * 4,
            num_workers = 1,
            pin_memory = True,
            sampler = sampler.CustomSampler(3, 4, augmented_dataset)
        )

        # Network that we're using in LFW dataset notebook
        # We cannot use identity, because now we're expecting that the outputs
        # of the network are vectors of size the embedding dimension
        # This way, we have a matrix of embedding vectors
        #
        # Using identity would produce nxnx3 (3 channels) outputs, making
        # `loss_functions.precompute_dict_of_classes` fail
        net = models.RandomNet(embedding_dimension = 4)

        # Get the metrics for a 1/5 of the training dataset
        intercluster_metrics = metrics.compute_intercluster_metrics(
            dataloader,
            net,
            int(len(augmented_dataset) * DATASET_PORTION)
        )

        # To check that the metrics were computed, just make some basic checks
        # All entries should be floats. Moreover, all should be greater than zero
        # So that is enough for our test
        self.assertGreater(intercluster_metrics["mean"], 0.0)
        self.assertGreater(intercluster_metrics["min"], 0.0)
        self.assertGreater(intercluster_metrics["max"], 0.0)
        self.assertGreater(intercluster_metrics["sd"], 0.0)


    def test_lfw_dataset_works_more_real(self):
        """
        I had problems running `compute_cluster_sizes_metrics`, and more
        specifically, in `__get_portion_of_dataset_and_embed`, for a portion of
        the LFW dataset. `test_lfw_dataset_works_basic` did not catch the error
        that now happens. We think that is because the network used there is not
        the network we're using in real training

        So this test is the same as `test_lfw_dataset_works_basic` but using
        the network that we use in the notebook
        """

        # Load the dataset
        transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(
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

        # Apply data augmentation for having at least 4 images per class
        augmented_dataset = data_augmentation.LazyAugmentatedDataset(
            base_dataset = dataset,
            min_number_of_images = 4,

            # Remember that the trasformation has to be random type
            # Otherwise, we could end with a lot of repeated images
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(250, 250)),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAutocontrast(),
            ])

        )

        # Now put a loader in front of the augmented dataset
        dataloader = torch.utils.data.DataLoader(
            augmented_dataset,
            batch_size = 3 * 4,
            num_workers = 2,
            pin_memory = True,
            sampler = sampler.CustomSampler(3, 4, augmented_dataset)
        )

        # Network that we're using in LFW dataset notebook
        # We cannot use identity, because now we're expecting that the outputs
        # of the network are vectors of size the embedding dimension
        # This way, we have a matrix of embedding vectors
        #
        # Using identity would produce nxnx3 (3 channels) outputs, making
        # `loss_functions.precompute_dict_of_classes` fail
        #
        # Permutation makes the test fail, because we're running on CPU
        # When running in GPU, `should_permute = True` is fine
        net = models.LFWResNet18(5)
        net.set_permute(should_permute = False)

        # Get the metrics for a 1/5 of the training dataset
        intercluster_metrics = metrics.compute_intercluster_metrics(
            dataloader,
            net,
            int(len(augmented_dataset) * DATASET_PORTION)
        )

        # To check that the metrics were computed, just make some basic checks
        # All entries should be floats. Moreover, all should be greater than zero
        # So that is enough for our test
        self.assertGreater(intercluster_metrics["mean"], 0.0)
        self.assertGreater(intercluster_metrics["min"], 0.0)
        self.assertGreater(intercluster_metrics["max"], 0.0)
        self.assertGreater(intercluster_metrics["sd"], 0.0)
