import unittest
import torch
import torchvision
import torchvision.transforms as transforms
import resource

import src.lib.metrics as metrics
import src.lib.utils as utils
import src.lib.data_augmentation as data_augmentation
import src.lib.sampler as sampler
import src.lib.models as models

# Precision that we want in certain tests
PLACES = 4

class TmpNetwork(torch.nn.Module):
    """Temporal network for this tests. RetrievalAdapter work with 4 mode tensors,
    that is to say, tensors with shape `[batch_size, channels, width, height]`

    We are going to pass, inside tensors of that shape, the embeddings that we
    want as a result, and therefore, doing so we can play with it in our tests,
    as we now the embedding output this network is going to produce
    """
    def __init__(self):
        super(TmpNetwork, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # We are expecting a batch of images, that is to say,
        # `x.shape == [batch_size, channels, width, height]`
        if len(x.shape) != 4:
            raise Exception(f"TMP class got a tensor with {len(x.shape)} modes, when we were expecting 4 modes (a batch of images)")

        # And we want to return a list of embeddings, that is to say,
        # `output.shape == [batch_size, 3]`
        # `squeeze` removes the dimensions with value 1. As we are passing tensors
        # with shape `[batch_size >= 1, 1, 3]`, we obtain `[batch_size, 3]`, which
        # is the size of the desired output embeddings
        output = x.squeeze()
        return output

class TestRetrievalAdapter(unittest.TestCase):

    def assertEqualsTensor(self, first: torch.Tensor, second: torch.Tensor):
        """
        Takes two tensors and checks if they are the same. If they are the same,
        it does nothing. If they are not the same, it raises an informative
        exception.

        So this function is meant to be used in unit tests, where raising an
        exception and not handling it is desired behaviour

        Two tensors are the same when they have the same dimensions, and each
        entry of the two tensors are the same
        """

        result = torch.equal(first, second)
        if result is False:
            msg = "Two given tensors are not the same!\n"
            msg = msg + f"First: {first}\n"
            msg = msg + f"Second: {second}\n"
            msg = msg + "\n"
            raise Exception(msg)

    def test_basic_case(self):

        # We use our TmpNetwork so we can now which is going to be the output
        # embedding, and thus, make checks based on that known behaviour
        net = TmpNetwork()

        # Wrap it with our adapter
        retrieval_net = models.RetrievalAdapter(net)

        # Put directly the embeddings in the query and candidates images as
        # we are using TmpNetwork
        query = torch.Tensor([[[0, 0, 0]]])
        candidates = torch.Tensor([
            [[[1, 0, 0]]],
            [[[0, 0, 0]]],
            [[[3, 0, 0]]],
            [[[2, 0, 0]]],
            [[[100, 100, 100]]]
        ])

        # Compute the best 3 candidates
        computed_best_candidates = retrieval_net.query(query, candidates, k = 3)
        expected_best_candidates = torch.Tensor([1, 0, 3])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

        # Compute the best 2 candidates
        computed_best_candidates = retrieval_net.query(query, candidates, k = 2)
        expected_best_candidates = torch.Tensor([1, 0])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

        # Compute the best candidate
        computed_best_candidates = retrieval_net.query(query, candidates, k = 1)
        expected_best_candidates = torch.Tensor([1])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

    def test_other_basic_case(self):

        # With the identity network we can pass as query and candidate images
        # directly the embeddings we want to play with
        net = TmpNetwork()

        # Wrap it with our adapter
        retrieval_net = models.RetrievalAdapter(net)

        # Put directly the embeddings in the query and candidates images as
        # we are using TmpNetwork
        query = torch.Tensor([[[0, 0, 0]]])
        candidates = torch.Tensor([
            [[[105, 0, 0]]],
            [[[110, 0, 0]]],
            [[[103, 0, 0]]],
            [[[102, 0, 0]]],
            [[[100, 0, 0]]]
        ])

        # Compute the best 3 candidates
        computed_best_candidates = retrieval_net.query(query, candidates, k = 3)
        expected_best_candidates = torch.Tensor([4, 3, 2])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

        # Compute the best 5 candidates
        computed_best_candidates = retrieval_net.query(query, candidates, k = 5)
        expected_best_candidates = torch.Tensor([4, 3, 2, 0, 1])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

    def test_boundaries_in_number_of_candidates(self):

        # We use our TmpNetwork so we can now which is going to be the output
        # embedding, and thus, make checks based on that known behaviour
        net = TmpNetwork()

        # Wrap it with our adapter
        retrieval_net = models.RetrievalAdapter(net)

        # Put directly the embeddings in the query and candidates images as
        # we are using TmpNetwork
        query = torch.Tensor([[[0, 0, 0]]])
        candidates = torch.Tensor([
            [[[1, 0, 0]]],
            [[[0, 0, 0]]],
            [[[3, 0, 0]]],
            [[[2, 0, 0]]],
            [[[100, 100, 100]]]
        ])

        # Compute the best 6 candidates
        # It should fail as we only have 5 candidates
        with self.assertRaises(ValueError):
            retrieval_net.query(query, candidates, k = 6)

    def test_basic_case_direct(self):

        # We are not going to use this network for the computations
        net = torch.nn.Identity()

        # Wrap it with our adapter
        retrieval_net = models.RetrievalAdapter(net)

        # Embeddings that we work with
        query_embedding = torch.Tensor([0, 0, 0])
        candidates_embeddings = torch.Tensor([
            [1, 0, 0],
            [0, 0, 0],
            [3, 0, 0],
            [2, 0, 0],
            [100, 100, 100]
        ])

        # Compute the best 3 candidates
        computed_best_candidates = retrieval_net.query_embedding(query_embedding, candidates_embeddings, k = 3)
        expected_best_candidates = torch.Tensor([1, 0, 3])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

        # Compute the best 2 candidates
        computed_best_candidates = retrieval_net.query_embedding(query_embedding, candidates_embeddings, k = 2)
        expected_best_candidates = torch.Tensor([1, 0])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

        # Compute the best candidate
        computed_best_candidates = retrieval_net.query(query_embedding, candidates_embeddings, k = 1)
        expected_best_candidates = torch.Tensor([1])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

    def test_other_basic_case_direct(self):

        # We are not going to use this network for the computations
        net = torch.nn.Identity()

        # Wrap it with our adapter
        retrieval_net = models.RetrievalAdapter(net)

        # Put directly the embeddings in the query and candidates images as
        # we are using TmpNetwork
        query_embedding = torch.Tensor([0, 0, 0])
        candidates_embeddings = torch.Tensor([
            [105, 0, 0],
            [110, 0, 0],
            [103, 0, 0],
            [102, 0, 0],
            [100, 0, 0]
        ])

        # Compute the best 3 candidates
        computed_best_candidates = retrieval_net.query_embedding(query_embedding, candidates_embeddings, k = 3)
        expected_best_candidates = torch.Tensor([4, 3, 2])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

        # Compute the best 5 candidates
        computed_best_candidates = retrieval_net.query_embedding(query_embedding, candidates_embeddings, k = 5)
        expected_best_candidates = torch.Tensor([4, 3, 2, 0, 1])
        self.assertEqualsTensor(computed_best_candidates, expected_best_candidates)

    def test_boundaries_in_number_of_candidates_direct(self):

        # We are not going to use this network for the computations
        net = torch.nn.Identity()

        # Wrap it with our adapter
        retrieval_net = models.RetrievalAdapter(net)

        # Embeddings we want to use
        query_embeddings = torch.Tensor([0, 0, 0])
        candidates_embeddings = torch.Tensor([
            [1, 0, 0],
            [0, 0, 0],
            [3, 0, 0],
            [2, 0, 0],
            [100, 100, 100]
        ])

        # Compute the best 6 candidates
        # It should fail as we only have 5 candidates
        with self.assertRaises(ValueError):
            retrieval_net.query_embedding(query_embeddings, candidates_embeddings, k = 6)
