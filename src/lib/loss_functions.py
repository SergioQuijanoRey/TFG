"""
Different loss functions used in the project
"""

import torch
import torch.nn as nn
from typing import List

# Bases for more complex loss functions
# ==================================================================================================

class TripletLoss(nn.Module):
    """
    Basic loss function that acts as the base for all batch loss functions

    This loss function is thought for single triplets. If you want to calculate the loss of a batch
    of triplets, use MeanTripletBatchTripletLoss
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:

        distance_positive = self.euclidean_distance(anchor, positive)
        distance_negative = self.euclidean_distance(anchor, negative)

        # Usamos Relu para que el error sea cero cuando la resta de las distancias
        # este por debajo del margen. Si esta por encima del margen, devolvemos la
        # identidad de dicho error. Es decir, aplicamos Relu a la formula que
        # tenemos debajo
        loss = torch.relu(distance_positive - distance_negative + self.margin)

        return loss

    def euclidean_distance(self, first: torch.Tensor, second: torch.Tensor) -> float:
        return ((first - second) * (first - second)).sum()


# Loss functions for batches of triplets
# ==================================================================================================

class MeanTripletBatchTripletLoss(nn.Module):
    """
    Computes the mean triplet loss of a batch of triplets
    Note that we are expecting a batch of triplets, and not a batch of images

    Thus, some offline mechanism for mining triplets is needed (ie. random triplets)
    """

    def __init__(self, margin=1.0):
        super(MeanTripletBatchTripletLoss, self).__init__()
        self.margin = margin
        self.base_loss = TripletLoss(self.margin)

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

    def precompute_negative_class(self, list_of_classes: List[List[int]]) -> List[List[int]]:
        """
        Computes a list of lists. Each list contains the positions of elements that are negative
        to the corresponding class
        ie. list_of_negatives[i] contains all elements whose class is not i-th class

        @param list_of_classes precomputed list of positions of classes
               This list is computed using precompute_list_of_classes
               For efficiency purpose
        """

        # Inicializamos la lista
        list_of_negatives = [None] * 10

        for label in range(10):
            list_of_negatives[label] = [
                idx
                for current_list in skip_i(list_of_classes, label)
                for idx in current_list
            ]

        return list_of_negatives


class BatchHardTripletLoss(nn.Module):
    """
    Implementation of Batch Hard Triplet Loss
    This loss function expects a batch of images, and not a batch of triplets
    """

    def __init__(self, margin=1.0):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.base_loss = TripletLoss(self.margin)

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
        self.list_of_classes = None

        # Si queremos usar self.list_of_classes para calcular todos los
        # negativos de una clase, necesitamos dos for que vamos a repetir
        # demasiadas veces
        self.list_of_negatives = None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        loss = 0

        # Pre-computamos la separacion en positivos y negativos
        self.list_of_classes = self.precomputations.precompute_list_of_classes(labels)

        # Pre-computamos la lista de negativos de cada clase
        self.list_of_negatives = self.precomputations.precompute_negative_class(self.list_of_classes)

        # Iteramos sobre todas los embeddings de las imagenes del dataset
        for embedding, img_label in zip(embeddings, labels):

            # Calculamos las distancias a positivos y negativos
            # Nos aprovechamos de la pre-computacion
            positive_distances = [
                self.base_loss.euclidean_distance(embedding, embeddings[positive])
                for positive in self.list_of_classes[img_label]
            ]

            # Ahora nos aprovechamos del segundo pre-computo realizado
            negative_distances = [
                self.base_loss.euclidean_distance(embedding, embeddings[negative])
                for negative in self.list_of_negatives
            ]

            # Tenemos una lista de tensores de un unico elemento (el valor
            # de la distancia). Para poder usar argmax pasamos todo esto
            # a un unico tensor
            positive_distances = torch.tensor(positive_distances)
            negative_distances = torch.tensor(negative_distances)

            # Calculamos la funcion de perdida
            positives = self.list_of_classes[img_label]
            negatives = self.list_of_negatives[img_label]

            worst_positive_idx = positives[torch.argmax(positive_distances)]
            worst_negative_idx = negatives[torch.argmin(negative_distances)]

            worst_positive = embeddings[worst_positive_idx]
            worst_negative = embeddings[worst_negative_idx]

            loss += self.base_loss(embedding, worst_positive, worst_negative)

        return loss

class BatchAllTripletLoss(nn.Module):
    """
    Implementation of Batch All Triplet Loss
    This loss function expects a batch of images, and not a batch of triplets
    """

    def __init__(self, margin=1.0):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.base_loss = TripletLoss(self.margin)

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
        self.list_of_classes = None

        # Si queremos usar self.list_of_classes para calcular todos los
        # negativos de una clase, necesitamos dos for que vamos a repetir
        # demasiadas veces
        self.list_of_negatives = None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:

        loss = 0

        # Pre-computamos la separacion en positivos y negativos
        self.list_of_classes = self.precomputations.precompute_list_of_classes(labels)

        # Pre-computamos la lista de negativos de cada clase
        self.list_of_negatives = self.precomputations.precompute_negative_class(self.list_of_classes)

        # Iteramos sobre todas los embeddings de las imagenes del dataset
        for embedding, img_label in zip(embeddings, labels):

            # Calculamos las distancias a positivos y negativos
            # Nos aprovechamos de la pre-computacion
            positive_distances = [
                self.base_loss.euclidean_distance(embedding, embeddings[positive])
                for positive in self.list_of_classes[img_label]
            ]

            # Ahora nos aprovechamos del segundo pre-computo realizado
            negative_distances = [
                self.base_loss.euclidean_distance(embedding, embeddings[negative])
                for negative in self.list_of_negatives
            ]

            # Tenemos una lista de tensores de un unico elemento (el valor
            # de la distancia). Para poder usar argmax pasamos todo esto
            # a un unico tensor
            positive_distances = torch.tensor(positive_distances)
            negative_distances = torch.tensor(negative_distances)

            # Calculamos la funcion de perdida
            positives = self.list_of_classes[img_label]
            negatives = self.list_of_negatives[img_label]

            worst_positive_idx = positives[torch.argmax(positive_distances)]
            worst_negative_idx = negatives[torch.argmin(negative_distances)]

            worst_positive = embeddings[worst_positive_idx]
            worst_negative = embeddings[worst_negative_idx]

            loss += self.base_loss(embedding, worst_positive, worst_negative)

        return loss
