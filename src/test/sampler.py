import unittest
import torch

import torchvision
import torchvision.datasets as datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets

# Para poder usar ResNet18 preentrenado
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from pprint import pprint
import gc
import functools
import math
import seaborn as sns

# Todas las piezas concretas que usamos de sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score
from sklearn.model_selection import ShuffleSplit

from typing import List

from src.lib.sampler import CustomSampler

# Global parameters
DATA_PATH = "data"
ONLINE_BATCH_SIZE = 32
NUM_WORKERS = 1

class TestCustomSampler(unittest.TestCase):
    def test_something(self):
        self.assertEqual(1, 2-1)

        # Transformaciones que queremos aplicar al cargar los datos
        # Ahora solo pasamos las imagenes a tensores, pero podriamos hacer aqui normalizaciones
        transform = transforms.Compose([
            transforms.ToTensor(),
            # TODO -- aqui podemos añadir la normaliazcion de datos
            ])

        # Cargamos el dataset usando torchvision, que ya tiene el conjunto
        # preparado para descargar
        train_dataset = torchvision.datasets.MNIST(
            root = DATA_PATH,
            train = True,
            download = True,
            transform = transform,
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = ONLINE_BATCH_SIZE,
            #shuffle = True,
            num_workers = NUM_WORKERS,
            pin_memory = True,
            # TODO -- using magic numbers
            sampler = CustomSampler(3, 16, train_dataset)
        )

        self.assertEqual(1,1)

