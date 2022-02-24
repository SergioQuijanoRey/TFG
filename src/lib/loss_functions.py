"""
Different loss functions used in the project
"""

import torch
import torch.nn as nn
from typing import List

class TripletLoss(nn.Module):
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

class TripletLossCustom(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLossCustom, self).__init__()
        self.margin = margin
        self.base_loss = TripletLoss(self.margin)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        losses = torch.tensor(
            [self.base_loss(current[0], current[1], current[2]) for current in batch],
            requires_grad=True
        )
        return losses.mean()

# Copiamos esto de https://stackoverflow.com/a/22279947
# Lo necesitamos para saltarnos el elemento de una lista de
# forma eficiente
import itertools as it
def skip_i(iterable, i):
    itr = iter(iterable)
    return it.chain(it.islice(itr, 0, i), it.islice(itr, 1, None))


class OnlineTripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.base_loss = TripletLoss(self.margin)

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
        self.list_of_classes = self.__precompute_list_of_classes(labels)

        # Pre-computamos la lista de negativos de cada clase
        self.list_of_negatives = self.__precompute_negative_class()

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

    def __precompute_list_of_classes(self, labels) -> List[List[int]]:
        """
        Calcula la lista con las listas de posiciones de cada clase por separado
        """

        # Inicializamos la lista de listas
        posiciones_clases = [[] for _ in range(10)]

        # Recorremos el dataset y colocamos los indices donde corresponde
        for idx, label in enumerate(labels):
            posiciones_clases[label].append(idx)

        return posiciones_clases

    def __precompute_negative_class(self):

        # Inicializamos la lista
        list_of_negatives = [None] * 10

        for label in range(10):
            list_of_negatives[label] = [
                idx
                for current_list in skip_i(self.list_of_classes, label)
                for idx in current_list
            ]

        return list_of_negatives
