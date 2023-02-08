"""
In this integration test we are going to check if training can complete
We're not going to check if the training produces a 'good' model, just that
the training process goes from start to end without errors

This integration test must be as close as `LFW Notebook.ipynb` as possible
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.datasets as datasets

# For using pre-trained ResNets
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime
from pprint import pprint
import gc
import functools
import math
import seaborn as sns
from collections import Counter
import time
import copy
import cProfile

import wandb

# All concrete pieces we're using form sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score
from sklearn.model_selection import ShuffleSplit


from tqdm import tqdm
from typing import List

# Now that files are loaded, we can import pieces of code
import src.lib.core as core
import src.lib.trainers as trainers
import src.lib.filesystem as filesystem
import src.lib.metrics as metrics
import src.lib.loss_functions as loss_functions
import src.lib.embedding_to_classifier as embedding_to_classifier
import src.lib.sampler as sampler
import src.lib.utils as utils
import src.lib.data_augmentation as data_augmentation

from src.lib.trainers import train_model_offline, train_model_online
from src.lib.train_loggers import SilentLogger, TripletLoggerOffline, TripletLoggerOnline, TrainLogger, CompoundLogger, IntraClusterLogger, InterClusterLogger
from src.lib.models import *
from src.lib.visualizations import *
from src.lib.models import ResNet18, LFWResNet18
from src.lib.loss_functions import MeanTripletBatchTripletLoss, BatchHardTripletLoss, BatchAllTripletLoss
from src.lib.embedding_to_classifier import EmbeddingToClassifier
from src.lib.sampler import CustomSampler
from src.lib.data_augmentation import AugmentatedDataset, LazyAugmentatedDataset


class IntegrationLFWDataset(unittest.TestCase):
    def test_lfw_dataset_trains(self):
        # Training parameters
        P = 2
        K = 2
        ONLINE_BATCH_SIZE = P * K
        TRAINING_EPOCHS = 1
        ONLINE_LEARNING_RATE = 0.01
        LOGGING_ITERATIONS = P * K * 20
        ONLINE_LOGGER_TRAIN_PERCENTAGE = 1 / 5
        ONLINE_LOGGER_VALIDATION_PERCENTAGE = 1 / 3
        NET_MODEL = "LFWResNet18"
        MARGIN = 1.0
        EMBEDDING_DIMENSION = 3
        BATCH_TRIPLET_LOSS_FUNCTION = "hard"
        USE_SOFTPLUS_LOSS = False
        USE_GT_ZERO_MEAN_LOSS = True
        LAZY_DATA_AUGMENTATION = True
        NUM_WORKERS = 1
        BASE_PATH = "./"
        DATA_PATH = os.path.join(BASE_PATH, "data")

        # Monkey patch the WANDB log
        def log_patch(msg: str):
            print(f"NOT LOGGIN TO WANDB: {msg}")

        wandb.log = log_patch

        # Load the dataset
        transform = transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5)
            ),
        ])

        train_dataset = torchvision.datasets.LFWPeople(
            root = DATA_PATH,
            split = "train",
            download = True,
            transform = transform,
        )

        test_dataset = torchvision.datasets.LFWPeople(
            root = DATA_PATH,
            split = "test",
            download = True,
            transform = transform,
        )

        # Use a really small portion of train dataset
        DATASET_PERCENTAGE = 0.01
        train_dataset, _ = core.split_train_test(train_dataset, DATASET_PERCENTAGE)
        train_dataset = train_dataset.dataset

        # Train / Validation split
        train_dataset, validation_dataset = core.split_train_test(train_dataset, 0.8)
        train_dataset = train_dataset.dataset
        validation_dataset = validation_dataset.dataset

        # Use custom sampler
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = ONLINE_BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory = True,
            sampler = CustomSampler(P, K, train_dataset)
        )

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size = ONLINE_BATCH_SIZE,
            shuffle = True,
            num_workers = NUM_WORKERS,
            pin_memory = True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size = ONLINE_BATCH_SIZE,
            shuffle = True,
            num_workers = NUM_WORKERS,
            pin_memory = True,
        )

        # Dataset augmentation
        AugmentationClass = LazyAugmentatedDataset if LAZY_DATA_AUGMENTATION is True else AugmentatedDataset

        train_dataset_augmented = AugmentationClass(
            base_dataset = train_dataset,
            min_number_of_images = K,

            # Remember that the trasformation has to be random type
            # Otherwise, we could end with a lot of repeated images
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=(250, 250)),
                transforms.RandomRotation(degrees=(0, 180)),
                transforms.RandomAutocontrast(),
            ])

        )

        train_loader_augmented = torch.utils.data.DataLoader(
            train_dataset_augmented,
            batch_size = ONLINE_BATCH_SIZE,
            num_workers = NUM_WORKERS,
            pin_memory = True,
            sampler = CustomSampler(P, K, train_dataset_augmented)
        )

        # Choose loss function
        batch_loss_function = None
        if BATCH_TRIPLET_LOSS_FUNCTION == "hard":
            batch_loss_function = BatchHardTripletLoss(MARGIN, use_softplus = USE_SOFTPLUS_LOSS, use_gt_than_zero_mean = USE_GT_ZERO_MEAN_LOSS)
        if BATCH_TRIPLET_LOSS_FUNCTION == "all":
            batch_loss_function = BatchAllTripletLoss(MARGIN, use_softplus = USE_SOFTPLUS_LOSS, use_gt_than_zero_mean =  USE_GT_ZERO_MEAN_LOSS)

        if batch_loss_function is None:
            raise Exception(f"BATCH_TRIPLET_LOSS global parameter got unexpected value: {BATCH_TRIPLET_LOSS_FUNCTION}")

        # Prepare the training
        net = None
        if NET_MODEL == "ResNet18":
            net = ResNet18(EMBEDDING_DIMENSION)
        elif NET_MODEL == "LightModel":
            net = LightModel(EMBEDDING_DIMENSION)
        elif NET_MODEL == "LFWResNet18":
            net = LFWResNet18(EMBEDDING_DIMENSION)
        else:
            raise Exception("Parameter 'NET_MODEL' has not a valid value")

        net.set_permute(False)

        parameters = dict()
        parameters["epochs"] = TRAINING_EPOCHS
        parameters["lr"] = ONLINE_LEARNING_RATE
        parameters["criterion"] = batch_loss_function

        print(net)

        # Prepare the loggers
        triplet_loss_logger = TripletLoggerOnline(
            net = net,
            iterations = LOGGING_ITERATIONS,
            loss_func = parameters["criterion"],
            train_percentage = ONLINE_LOGGER_TRAIN_PERCENTAGE,
            validation_percentage = ONLINE_LOGGER_VALIDATION_PERCENTAGE,
            greater_than_zero = USE_GT_ZERO_MEAN_LOSS,
        )

        cluster_sizes_logger = IntraClusterLogger(
            net = net,
            iterations = LOGGING_ITERATIONS,
            train_percentage = ONLINE_LOGGER_TRAIN_PERCENTAGE,
            validation_percentage = ONLINE_LOGGER_VALIDATION_PERCENTAGE,
        )

        intercluster_metrics_logger = InterClusterLogger(
            net = net,
            iterations = LOGGING_ITERATIONS,
            train_percentage = ONLINE_LOGGER_TRAIN_PERCENTAGE,
            validation_percentage = ONLINE_LOGGER_VALIDATION_PERCENTAGE,
        )

        logger = CompoundLogger([
            triplet_loss_logger,
            cluster_sizes_logger,
            intercluster_metrics_logger
        ])

        # Run the training loop
        ts = time.time()
        training_history = train_model_online(
            net = net,
            path = os.path.join(BASE_PATH, "tmp"),
            parameters = parameters,
            train_loader = train_loader_augmented,
            validation_loader = validation_loader,
            name = NET_MODEL,
            logger = logger,
            snapshot_iterations = None
        )

        # Compute how long it took
        te = time.time()
        print(f"It took {te - ts} seconds to train")
