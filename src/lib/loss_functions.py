"""
Different loss functions used in the project
"""

import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable
from typing import List, Tuple, Dict

# utils import depend on enviroment (local or remote), so we can do two try-except blocks
# for dealing with that
try:
    import utils
except Exception as e:
    pass
try:
    import src.lib.utils as utils
except Exception as e:
    pass

# Bases for more complex loss functions
# ==================================================================================================

def distance_function(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """
    Basic distance function. It's the base for all losses implemented in this module
    """
    return ((first - second) * (first - second)).sum().sqrt()


class TripletLoss(nn.Module):
    """
    Basic loss function that acts as the base for all batch loss functions

    This loss function is thought for single triplets. If you want to calculate the loss of a batch
    of triplets, use MeanTripletBatchTripletLoss
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> float:

        distance_positive = distance_function(anchor, positive)
        distance_negative = distance_function(anchor, negative)

        # Usamos Relu para que el error sea cero cuando la resta de las distancias
        # este por debajo del margen. Si esta por encima del margen, devolvemos la
        # identidad de dicho error. Es decir, aplicamos Relu a la formula que
        # tenemos debajo
        return self.loss_from_distances(distance_positive, distance_negative)

    def loss_from_distances(self, positive_distance: float, negative_distance: float) -> float:
        """
        Compute the loss using the using the pre-computed distances
        @param positive_distance the distance among anchor and positive
        @param negative_distance the distance among anchor and negative
        """

        # We use ReLU to utilize the pytorch to compute max(0, val) used in the triplet loss
        return torch.relu(positive_distance - negative_distance + self.margin)


class SoftplusTripletLoss(nn.Module):
    """
    Slight modification of the basic loss function that acts as the base for all batch loss functions
    Instead of using [• + m]_+, use softplus ln(1 + exp(•))

    This loss function is thought for single triplets. If you want to calculate the loss of a batch
    of triplets, use MeanTripletBatchTripletLoss
    """

    def __init__(self):
        super(SoftplusTripletLoss, self).__init__()
        self.softplus = nn.Softplus(beta = 1, threshold = 1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> float:

        distance_positive = distance_function(anchor, positive)
        distance_negative = distance_function(anchor, negative)

        # Usamos Relu para que el error sea cero cuando la resta de las distancias
        # este por debajo del margen. Si esta por encima del margen, devolvemos la
        # identidad de dicho error. Es decir, aplicamos Relu a la formula que
        # tenemos debajo
        return self.loss_from_distances(distance_positive, distance_negative)

    def loss_from_distances(self, positive_distance: float, negative_distance: float) -> float:
        """
        Compute the loss using the using the pre-computed distances
        @param positive_distance the distance among anchor and positive
        @param negative_distance the distance among anchor and negative
        """

        # We use ReLU to utilize the pytorch to compute max(0, val) used in the triplet loss
        return self.softplus(positive_distance - negative_distance)

# Loss functions for batches of triplets
# ==================================================================================================

class MeanTripletBatchTripletLoss(nn.Module):
    """
    Computes the mean triplet loss of a batch of triplets
    Note that we are expecting a batch of triplets, and not a batch of images

    Thus, some offline mechanism for mining triplets is needed (ie. random triplets)
    """

    def __init__(self, margin=1.0, use_softplus = False):
        super(MeanTripletBatchTripletLoss, self).__init__()
        self.margin = margin
        self.base_loss = TripletLoss(self.margin) if use_softplus is False else SoftplusTripletLoss()

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        losses = torch.tensor(
            [self.base_loss(current[0], current[1], current[2]) for current in batch],
            requires_grad=True
        )
        return losses.mean()

# Loss functions for batches of images (and not triplets)
# ==================================================================================================

# Copiamos esto de https://stackoverflow.com/a/22279947
# Lo necesitamos para saltarnos el elemento de una lista de
# forma eficiente
import itertools as it
def skip_i(iterable, i):
    itr = iter(iterable)
    return it.chain(it.islice(itr, 0, i), it.islice(itr, 1, None))


class BatchBaseTripletLoss(nn.Module):
    """
    Class containing some shared code to all Batch Triplet Loss Variants

    Thus, this class should only used in certain implementations of loss functions, and not directly
    by the user

    At the moment, all of shared code correspond to pre-computation code
    """

    def __init__(self):
        pass

    def precompute_list_of_classes(self, labels) -> List[List[int]]:
        """
        Computes a list containing list. Each list contains the positions of elements of given class
        ie. class_positions[i] contains all positions of elements of i-th class
        """

        # Inicializamos la lista de listas
        class_positions = [[] for _ in range(10)]

        # Recorremos el dataset y colocamos los indices donde corresponde
        for idx, label in enumerate(labels):
            class_positions[label].append(idx)

        return class_positions

    def precompute_negative_class(self, dict_of_classes: Dict[int, List[int]]) -> Dict[int, List[int]]:
        """
        Computes a dictionary `dict_of_negatives`. Each key i has associated a list with all the
        indixes of elements of other class.

        For example, `dict_of_negatives[4]` has all the indixes of elements whose class is not 4

        @param dict_of_classes precomputed dict of positions of classes, computed using `utils.precompute_dict_of_classes`
        @returns `dict_of_negatives` a dict as described before
        """

        # Inicializamos la lista
        dict_of_negatives = dict()

        # Iterate over all classes present in the dataset
        # The labels are enconded as the keys of `dict_of_classes`
        for label in dict_of_classes.keys():

            dict_of_negatives[label] = []
            for other_label in dict_of_classes.keys():

                if other_label == label:
                    continue

                dict_of_negatives[label] = dict_of_negatives[label] + dict_of_classes[other_label]

        return dict_of_negatives

    def precompute_pairwise_distances(self, embeddings: torch.Tensor, distance_function) -> Dict[Tuple[int, int], float]:
        """
        Given a batch of embeddings and a distance function, precomputes all the pairwise distances.
        @return distances, dict of distances where distances[i][j] = distance(x_i, x_j)
                Only half of the matrix is computed, distances[i][j] where i <= j
        """

        distances = dict()

        for first_idx, first in enumerate(embeddings):
            for second_idx, second in enumerate(embeddings):

                # Only half of the matrix is computed
                if first_idx > second_idx:
                    continue

                # Store this distance
                distances[first_idx, second_idx] = distance_function(first, second)

        return distances


class BatchHardTripletLoss(nn.Module):
    """
    Implementation of Batch Hard Triplet Loss
    This loss function expects a batch of images, and not a batch of triplets

    Large minibatches are encouraged, as we are computing, for each img in the minibatch, its hardest
    positive and negative. So having a large minibatch makes more likely that a non-trivial positive
    and negative get found in the minibatch

    In this class we pre-compute all pairwise distances. In this case is worth the overhead because
    we need to constantly compute all positive all negative distances.
    """

    def __init__(self, margin: float = 1.0, use_softplus = False, use_gt_than_zero_mean = False):
        super(BatchHardTripletLoss, self).__init__()

        self.margin = margin

        # Select base loss depending on given parameters
        self.base_loss = TripletLoss(self.margin) if use_softplus is False else SoftplusTripletLoss()

        # Select if all summands have to taken in account to compute the mean or only those sumamnds
        # greater than zero
        # This is useless when using softplus loss function
        self.use_gt_than_zero_mean = use_gt_than_zero_mean

        # Class to access shared code across all Batch Triplet Loss functions
        self.precomputations = BatchBaseTripletLoss()

        # Pre-computamos una lista de listas en la que accedemos a los
        # elementos de la forma list[label][posicion]
        # Con esto nos evitamos tener que realizar la separacion en positivos
        # y negativos repetitivamente
        #
        # Notar que el pre-computo debe realizarse por cada llamada a forward,
        # con el minibatch correspondiente. Por tanto, nos beneficia usar minibatches
        # grandes
        self.dict_of_classes = None

        # Si queremos usar self.list_of_classes para calcular todos los
        # negativos de una clase, necesitamos dos for que vamos a repetir
        # demasiadas veces
        self.list_of_negatives = None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:

        loss = 0

        # Pre-computamos la separacion en positivos y negativos
        self.dict_of_classes = utils.precompute_dict_of_classes([int(label) for label in labels])

        # Pre-computamos la lista de negativos de cada clase
        self.list_of_negatives = self.precomputations.precompute_negative_class(self.dict_of_classes)

        # Count non zero losses in order to compute the > 0 mean
        non_zero_losses = 0

        # Iteramos sobre todas los embeddings de las imagenes del dataset
        # TODO -- try to use pre-computed pairwise distances and see if it speeds up the calculation
        for embedding, img_label in zip(embeddings, labels):

            # Calculamos las distancias a positivos y negativos
            # Nos aprovechamos de la pre-computacion
            positive_distances = [
                distance_function(embedding, embeddings[positive])
                for positive in self.dict_of_classes[int(img_label)]
            ]

            # Ahora nos aprovechamos del segundo pre-computo realizado
            negative_distances = [
                distance_function(embedding, embeddings[negative])
                for negative in self.list_of_negatives[int(img_label)]
            ]

            # Tenemos una lista de tensores de un unico elemento (el valor
            # de la distancia). Para poder usar argmax pasamos todo esto
            # a un unico tensor
            positive_distances = torch.tensor(positive_distances)
            negative_distances = torch.tensor(negative_distances)

            # Calculamos la funcion de perdida
            positives = self.dict_of_classes[int(img_label)]
            negatives = self.list_of_negatives[int(img_label)]

            worst_positive_idx = positives[torch.argmax(positive_distances)]
            worst_negative_idx = negatives[torch.argmin(negative_distances)]

            worst_positive = embeddings[worst_positive_idx]
            worst_negative = embeddings[worst_negative_idx]

            curr_loss = self.base_loss(embedding, worst_positive, worst_negative)
            loss += curr_loss

            if curr_loss > 0:
                non_zero_losses += 1

        # Keep track of active triplets
        wandb.log({"Non zero losses": non_zero_losses})
        wandb.log({"Non zero losses (%)": non_zero_losses / len(labels) * 100.0})

        # Return the mean of the loss
        # Compute the mean depending on self.use_gt_than_zero_mean
        mean = None
        if self.use_gt_than_zero_mean is True:
            mean = loss / non_zero_losses
        else:
            mean = loss / len(labels)

        return mean


class BatchAllTripletLoss(nn.Module):
    """
    Implementation of Batch All Triplet Loss
    This loss function expects a batch of images, and not a batch of triplets

    In order to use this loss function, minibatches should not be very large. Otherwise, we can run
    out of RAM. That is not the case for BatchHardTripletLoss, where large minibatches are encouraged
    """

    def __init__(self, margin=1.0, use_softplus = False, use_gt_than_zero_mean = False):
        super(BatchAllTripletLoss, self).__init__()
        self.margin = margin

        # Select the base loss depending on given parameters
        self.base_loss = TripletLoss(self.margin) if use_softplus is False else SoftplusTripletLoss()

        # Class to access shared code across all Batch Triplet Loss functions
        self.precomputations = BatchBaseTripletLoss()

        # Select if all summands have to taken in account to compute the mean or only those sumamnds
        # greater than zero
        # This is useless when using softplus loss function
        self.use_gt_than_zero_mean = use_gt_than_zero_mean

        # Pre-computamos una lista de listas en la que accedemos a los
        # elementos de la forma list[label][posicion]
        # Con esto nos evitamos tener que realizar la separacion en positivos
        # y negativos repetitivamente
        #
        # Notar que el pre-computo debe realizarse por cada llamada a forward,
        # con el minibatch correspondiente. Por tanto, nos beneficia usar minibatches
        # grandes
        # TODO -- deprecated docs!
        self.dict_of_classes: Dict[int, List[int]] = None

        # Si queremos usar self.list_of_classes para calcular todos los
        # negativos de una clase, necesitamos dos for que vamos a repetir
        # demasiadas veces
        self.list_of_negatives = None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> float:

        loss = 0

        # Precomputations to speed up calculations
        self.dict_of_classes = utils.precompute_dict_of_classes([int(label) for label in labels])
        self.list_of_negatives = self.precomputations.precompute_negative_class(self.dict_of_classes)
        self.pairwise_distances = self.precomputations.precompute_pairwise_distances(embeddings, distance_function)

        # For computing the mean
        # We have to instantiate the var using this syntax so backpropagation can be done properly
        summands_used = Variable(torch.tensor(0.0), requires_grad = True)

        # For logging purposes, see how many summands in total are seen
        seen_summands = 0

        # Iterate over all elements, that act as anchors
        for [anchor_idx, _], anchor_label in zip(enumerate(embeddings), labels):

            # Compute all combinations of positive / negative loss
            # Use the precomputed distances to speed up this calculation
            losses = torch.tensor([
                self.base_loss.loss_from_distances(
                    self.pairwise_distances[self.__resort_dict_idx(anchor_idx, positive_idx)],
                    self.pairwise_distances[self.__resort_dict_idx(anchor_idx, negative_idx)]
                )
                for positive_idx in self.dict_of_classes[int(anchor_label)] if positive_idx != anchor_idx
                for negative_idx in self.list_of_negatives[int(anchor_label)]
            ])

            # Accumulate loss
            loss += torch.sum(losses)

            # Compute summands used, depending if we're counting all summands or only > 0 summands
            if self.use_gt_than_zero_mean is True:
                summands_used = summands_used + torch.count_nonzero(losses)
            else:
                summands_used = summands_used + len(losses)

            # In both cases, the seen summands are the same
            # Again, this is used for logging purposes
            seen_summands += len(losses)


        # Keep track of active triplets
        wandb.log({"Non zero losses": summands_used})
        wandb.log({"Non zero losses (%)": summands_used / seen_summands * 100.0})

        # Return the mean of the loss
        # Summands used depend on self.use_gt_than_zero_mean
        return loss / summands_used

    def __resort_dict_idx(self, first: int, second: int) -> Tuple[int, int]:
        """Our dict containing pre-computed distances only has entries indexed by [i, j] where i <= j"""

        if first > second:
            return second, first

        return first, second
