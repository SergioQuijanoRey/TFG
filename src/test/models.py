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
        if result == False:
            msg = "Two given tensors are not the same!\n"
            msg = msg + f"First: {first}\n"
            msg = msg + f"Second: {second}\n"
            msg = msg + "\n"
            raise Exception(msg)



    def test_basic_case(self):

        # With the identity network we can pass as query and candidate images
        # directly the embeddings we want to play with
        net = torch.nn.Identity()

        # Wrap it with our adapter
        retrieval_net = models.RetrievalAdapter(net)

        # Put directly the embeddings in the query and candidates images as
        # we are using an Identity network
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
