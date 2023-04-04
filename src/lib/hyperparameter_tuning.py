import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import ShuffleSplit

import src.lib.metrics as metrics

from typing import Callable

# TODO -- translate to english
# TODO -- document the parameters
def custom_cross_validation(
    train_dataset: Dataset,
    k: int,
    random_seed: int,
    network_trainer: Callable[[DataLoader], torch.nn.Module],
    loader_generator: Callable[[Dataset], DataLoader],
    loss_function: Callable[[torch.nn.Module, DataLoader], float]
):
    """
    Perform k-fold cross validation

    `network_trainer` should be a function that produces a network trained on
    a given dataloader for each fold (see `loader_generator`). It has to return
    the trained network

    `loader_generator` should be a function that takes one fold (which is a
    dataset) and converts it to a `DataLoader`. For example, taking care of
    creating one of our custom dataloaders

    `loss_function` should be a function that takes a trained network and
    the validation fold dataloader, and produces the loss value that we want
    to optimize
    """

    # Definimos la forma en la que vamos a hacer el split de los folds
    ss = ShuffleSplit(n_splits=k, test_size=0.25, random_state=random_seed)

    # Lista en la que guardamos las perdidas encontradas en cada fold
    losses = []

    # Iteramos usando el split que nos da sklearn
    for train_index, validation_index in ss.split(train_dataset):

        # Tenemos los indices de los elementos, asi que tomamos los dos datasets
        # usando dichos indices
        train_fold = [train_dataset[idx] for idx in train_index]
        validation_fold = [train_dataset[idx] for idx in validation_index]

        # Transform dataset folds to dataloaders
        train_loader = loader_generator(train_fold)
        validation_loader = loader_generator(validation_fold)

        # Train the network and get the trained network
        net = network_trainer(train_loader)

        # Evaluamos la red en el fold de validacion
        net.eval()
        loss = loss_function(net, validation_loader)

        # Añadimos el loss a nuestra lista
        losses.append(loss)

    # Devolvemos el array en formato numpy para que sea mas comodo trabajar con ella
    return np.array(losses)
