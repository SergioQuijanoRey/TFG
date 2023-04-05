# Labeled Faces in the Wild (LFW)
# ==============================================================================
#
# - From the Notebook used for *MNIST dataset*, we try to solve identification for *Labeled Faces in the Wild dataset*
#     - We're loading the data from the pytorch implementation of this dataset
#     - But more information about this dataset can be found [the official website](http://vis-www.cs.umass.edu/lfw/)
#     - In that website, you can download the dataset in its original form
# - **Main differences from MNIST notebook**
#     - Obviously, the dataset is not the same
#     - Also, as we're solving a much harder dataset, and online triplet loss has been tested in *MNIST notebook*, we get rid of random triplets experiments

# Global Parameters of the Notebook
# ==============================================================================

## Paths
# ==============================================================================
#
# - Parameters related to data / model / lib paths

# Lib to define paths
import os

# Define if we are running the notebook in our computer ("local")
# or in Google Colab ("remote")
# or in UGR's server ("ugr")
RUNNING_ENV = "ugr"

# Base path for the rest of paths defined in the notebook
BASE_PATH = None
if RUNNING_ENV == "local":
    BASE_PATH = "./"
elif RUNNING_ENV == "remote":
    BASE_PATH = "/content/drive/MyDrive/Colab Notebooks/"
elif RUNNING_ENV == "ugr":
    BASE_PATH = "/mnt/homeGPU/squijano/TFG/"
else:
    raise Exception(f"RUNNING ENV is not valid, got value {RUNNING_ENV}")

# Path to our lib dir
LIB_PATH = os.path.join(BASE_PATH, "lib")

# Path where we store training / test data
DATA_PATH = os.path.join(BASE_PATH, "data")

# Dir with all cached models
# This cached models can be loaded from disk when training is skipped
MODEL_CACHE_FOLDER = os.path.join(BASE_PATH, "cached_models")

# Cache for the augmented dataset
AUGMENTED_DATASET_CACHE_FILE = os.path.join(BASE_PATH, "cached_augmented_dataset.pt")

# File where the logs are written
LOGGING_FILE = os.path.join(BASE_PATH, "training.log")

# Binary file where the stats of the profiling are saved
PROFILE_SAVE_FILE = os.path.join(BASE_PATH, "training_profile.stat")


## ML parameters
# ==============================================================================
#
# - Parameters related to machine learning
# - For example, batch sizes, learning rates, ...


# Parameters of P-K sampling
P = 100   # Number of classes used in each minibatch
K = 2     # Number of images sampled for each selected class

# Batch size for online training
# We can use `P * K` as batch size. Thus, minibatches will be
# as we expect in P-K sampling.
#
# But we can use `n * P * K`. Thus, minibatches will be n P-K sampling
# minibatche concatenated together
# Be careful when doing this because it can be really slow, and there is no
# clear reason to do this
ONLINE_BATCH_SIZE = P * K

# Epochs for hard triplets, online training
TRAINING_EPOCHS = 25

# Learning rate for hard triplets, online training
ONLINE_LEARNING_RATE = 0.0001

# How many single elements we want to see before logging
# It has to be a multiple of P * K, otherwise `should_log` would return always
# false as `it % LOGGING_ITERATIONS != 0` always
#
# `LOGGING_ITERATIONS = P * K * n` means we log after seeing `n` P-K sampled
# minibatches
LOGGING_ITERATIONS = P * K * 100

# Which percentage of the training and validation set we want to use for the logging
ONLINE_LOGGER_TRAIN_PERCENTAGE = 1 / 10
ONLINE_LOGGER_VALIDATION_PERCENTAGE = 1 / 3

# Choose which model we're going to use
# Can be "ResNet18", "LightModel", "LFWResNet18" or "LFWLightModel"
NET_MODEL = "LFWResNet18"

# Epochs used in k-Fold Cross validation
# k-Fold Cross validation used for parameter exploration
HYPERPARAMETER_TUNING_EPOCHS = 10

# Number of folds used in k-fold Cross Validation
NUMBER_OF_FOLDS = 4

# Margin used in the loss function
MARGIN = 1.0

# Dim of the embedding calculated by the network
EMBEDDING_DIMENSION = 5

# Number of neighbours considered in K-NN
# K-NN used for transforming embedding task to classification task
NUMBER_NEIGHBOURS = 4

# Batch Triplet Loss Function
# This way we can choose among "hard", "all"
BATCH_TRIPLET_LOSS_FUNCTION = "hard"

# Whether or not use softplus loss function instead of vanilla triplet loss
USE_SOFTPLUS_LOSS = True

# Count all sumamnds in the mean loss or only those summands greater than zero
USE_GT_ZERO_MEAN_LOSS = True

# Wether or not use lazy computations in the data augmentation
LAZY_DATA_AUGMENTATION = True

# Where or not add penalty term to the loss function
ADD_NORM_PENALTY = False

# If we add that penalty term, which scaling factor to use
PENALTY_FACTOR = MARGIN

# If we want to wrap our model into a normalizer
# That wrapper divides each vector by its norm, thus, forcing norm 1 on each vector
NORMALIZED_MODEL_OUTPUT = True

# If its None, we do not perform gradient clipping
# If its a Float value, we perform gradient clipping, using that value as a
# parameter for the max norm
GRADIENT_CLIPPING = MARGIN * 10


## Section parameters
# ==============================================================================

# - Flags to choose if some sections will run or not
# - This way we can skip some heavy computations when not needed


# Skip hyper parameter tuning for online training
SKIP_HYPERPARAMTER_TUNING = False

# Skip training and use a cached model
# Useful for testing the embedding -> classifier transformation
# Thus, when False training is not computed and a cached model
# is loaded from disk
# Cached models are stored in `MODEL_CACHE_FOLDER`
USE_CACHED_MODEL = False

# Skip data augmentation and use the cached augmented dataset
USE_CACHED_AUGMENTED_DATASET = False

# Most of the time we're not exploring the data, but doing
# either hyperparameter settings or training of the model
# So if we skip this step we can start the process faster
SKIP_EXPLORATORY_DATA_ANALYSYS = True

# Wether or not profile the training
# This should be False most of the times
# Note that profiling adds a significant overhead to the training
PROFILE_TRAINING = False


## WANDB Parameters
# ==============================================================================

from datetime import datetime

# Name for the project
# One project groups different runs
WANDB_PROJECT_NAME = "Labeled Faces in the Wild (LFW) Dataset"

# Name for this concrete run
# I don't care too much about it, because wandb tracks the parameters we use
# in this run (see "Configuration for Weights and Biases" section)
WANDB_RUN_NAME = str(datetime.now())


## Others
# ==============================================================================

# Number of workers we want to use
# We can have less, equal or greater num of workers than CPUs
# In the following forum:
#   https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
# they recomend to explore this parameter, growing it until system RAM saturates
# Using a value greater than 2 makes pytorch tell us that this value is not optimal
# So sticking with what pytorch tells uss
NUM_WORKERS = 2

# Fix random seed to make reproducible results
RANDOM_SEED = 123456789

# Add some paths to PYTHONPATH
# ==============================================================================

# Python paths are difficult to manage
# In this script we can do something like:
# `import lib.core as core` and that's fine
# But in lib code we cannot import properly the modules

import sys
sys.path.append(os.path.join(BASE_PATH, "src"))
sys.path.append(os.path.join(BASE_PATH, "src/lib"))
sys.path.append(BASE_PATH)

# Importing the modules we are going to use
# ==============================================================================

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
import dotenv

# All concrete pieces we're using form sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score


from tqdm import tqdm
from typing import List

# Now that files are loaded, we can import pieces of code
import lib.core as core
import lib.trainers as trainers
import lib.filesystem as filesystem
import lib.metrics as metrics
import lib.loss_functions as loss_functions
import lib.embedding_to_classifier as embedding_to_classifier
import lib.sampler as sampler
import lib.utils as utils
import lib.data_augmentation as data_augmentation
import lib.split_dataset as split_dataset
import lib.hyperparameter_tuning as hptuning

from lib.trainers import train_model_offline, train_model_online
from lib.train_loggers import SilentLogger, TripletLoggerOffline, TripletLoggerOnline, TrainLogger, CompoundLogger, IntraClusterLogger, InterClusterLogger
from lib.models import *
from lib.visualizations import *
from lib.models import ResNet18, LFWResNet18, LFWLightModel, NormalizedNet
from lib.loss_functions import MeanTripletBatchTripletLoss, BatchHardTripletLoss, BatchAllTripletLoss, AddSmallEmbeddingPenalization
from lib.embedding_to_classifier import EmbeddingToClassifier
from lib.sampler import CustomSampler
from lib.data_augmentation import AugmentatedDataset, LazyAugmentatedDataset


# Configuration of the logger
# ==============================================================================
#
# - Here we set the configuration for all logging done
# - In lib, `logging.getLogger("MAIN_LOGGER")` is used everywhere, so we get it, configure it once, and use that config everywhere


# Get the logger that is used everywhere
file_logger = logging.getLogger("MAIN_LOGGER")

# Configure it
file_logger.propagate = False # Avoid propagatint to upper logger, which logs to
                         # the console
file_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s::%(levelname)s::%(funcName)s::> %(message)s")
file_handler = logging.FileHandler(LOGGING_FILE)
file_handler.setFormatter(formatter)
file_logger.addHandler(file_handler)

# 'application' code
file_logger.debug('debug message')


# Configuration for Weigths and Biases
# ==============================================================================
#
# - We're going to use `wandb` for tracking the training of the models
# - In this section, we configure `wandb`, mainly selecting which parameters of the notebook are we going to track


# Select which parameters of the notebook we're going to track in wand
# This has to be done before `wandb.init()` in order to pass this dict to
# `wandb.init`
#
# I could create a config dict in "Global Parameters of the Notebook" and pass it
# rightaway. Or use directly wandb.config.SOMETHING everywhere. We don't do this
# because of the following reasons:
#
# 1. We don't want to track all parameters (ie. section parameters, dir paths...)
# 2. At this moment, we're not 100% sure that wandb is the right tool, so we are
#    looking for loose coupling

wandb_config_dict = {}


wandb_config_dict["P"] = P
wandb_config_dict["K"] = K
wandb_config_dict["ONLINE_BATCH_SIZE"] = ONLINE_BATCH_SIZE
wandb_config_dict["TRAINING_EPOCHS"] = TRAINING_EPOCHS
wandb_config_dict["ONLINE_LEARNING_RATE"] = ONLINE_LEARNING_RATE
wandb_config_dict["LOGGING_ITERATIONS"] = LOGGING_ITERATIONS
wandb_config_dict["ONLINE_LOGGER_TRAIN_PERCENTAGE"] = ONLINE_LOGGER_TRAIN_PERCENTAGE
wandb_config_dict["ONLINE_LOGGER_VALIDATION_PERCENTAGE"] = ONLINE_LOGGER_VALIDATION_PERCENTAGE
wandb_config_dict["NET_MODEL"] = NET_MODEL
wandb_config_dict["HYPERPARAMETER_TUNING_EPOCHS"] = HYPERPARAMETER_TUNING_EPOCHS
wandb_config_dict["NUMBER_OF_FOLDS"] = NUMBER_OF_FOLDS
wandb_config_dict["MARGIN"] = MARGIN
wandb_config_dict["EMBEDDING_DIMENSION"] = EMBEDDING_DIMENSION
wandb_config_dict["NUMBER_NEIGHBOURS"] = NUMBER_NEIGHBOURS
wandb_config_dict["BATCH_TRIPLET_LOSS_FUNCTION"] = BATCH_TRIPLET_LOSS_FUNCTION
wandb_config_dict["USE_SOFTPLUS_LOSS"] = USE_SOFTPLUS_LOSS
wandb_config_dict["USE_GT_ZERO_MEAN_LOSS"] = USE_GT_ZERO_MEAN_LOSS
wandb_config_dict["PROFILE_TRAINING"] = PROFILE_TRAINING
wandb_config_dict["ADD_NORM_PENALTY"] = ADD_NORM_PENALTY
wandb_config_dict["PENALTY_FACTOR"] = PENALTY_FACTOR
wandb_config_dict["NORMALIZED_MODEL_OUTPUT"] = NORMALIZED_MODEL_OUTPUT
wandb_config_dict["GRADIENT_CLIPPING"] = GRADIENT_CLIPPING

# If we're running in UGR's servers, we need to set some ENV vars
# Otherwise, wandb is going to write to dirs that it has no access
# Also, pytorch tries to save pretrained models in the home folder
if RUNNING_ENV == "ugr":

    print("-> Changing dir env values")
    utils.change_dir_env_vars(base_path = BASE_PATH)
    print("-> Changing done!")
    print("")

    print("-> Login again to WANDB")
    utils.login_wandb()
    print("-> Login done!")
    print("")


# Init the wandb tracker
# We need to do this before
#  wandb.login()
wandb.init(
    project = WANDB_PROJECT_NAME,
    name = WANDB_RUN_NAME,
    config = wandb_config_dict,
)

# Functions that we are going to use
# ==============================================================================


def show_learning_curve(training_history: dict):
    # Take two learning curves
    loss = training_history['loss']
    val_loss = training_history['val_loss']

    # Move the lists to cpu, because that's what matplotlib needs
    loss = [loss_el.cpu() for loss_el in loss]
    val_loss = [val_loss_el.cpu() for val_loss_el in val_loss]

    # Show graphics
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()

def try_to_clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


# Importing and preparing the data
# ==============================================================================

## Dataset loading
# ==============================================================================
#
# - As mentioned before, we're using the `pytorch` implementation for this dataset


# Transformations that we want to apply when loading the data
# Now we are only transforming images to tensors (pythorch only works with tensors)
# But we can apply here some normalizations
transform = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize(
         (0.5, 0.5, 0.5),
         (0.5, 0.5, 0.5)
     ),
])

# Load the dataset
# torchvision has a method to download and load the dataset
# TODO -- look what's the difference between this dataset and LFWPairs
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

# Train -> train / validation split
# This function returns a `WrappedSubset` so we can still have access to
# `targets` attribute. With `Subset` pytorch class we cannot do that
train_dataset, validation_dataset = split_dataset.split_dataset(train_dataset, 0.8)


## Use our custom sampler
# ==============================================================================
#
# - We want to use custom `Sampler` to do $P-K$ sampling
#     - For each minibatch, select $P$ random classes. For each selected class, select $K$ random images
# - Thus, each minibatch has a size of $P \times K$


# New dataloaders that use our custom sampler
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = ONLINE_BATCH_SIZE,
    num_workers = NUM_WORKERS,
    pin_memory = True,
    sampler = CustomSampler(P, K, train_dataset)
)

# TODO -- here I don't know if use default sampler or custom sampler
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size = ONLINE_BATCH_SIZE,
    shuffle = True,
    num_workers = NUM_WORKERS,
    pin_memory = True,
)

# TODO -- here I don't know if use default sampler or custom sampler
test_loader = torch.utils.data.DataLoader(
  test_dataset,
  batch_size = ONLINE_BATCH_SIZE,
  shuffle = True,
  num_workers = NUM_WORKERS,
  pin_memory = True,
)


# Data augmentation
# ==============================================================================
#
# - As we've seen before, the main problem with this dataset is that most of the classes have only one or two images associated
# - So we're going to apply data augmentation to have at least a minimun number of images per class
#
# **Alternatives to do this**:
#
# 1. Use `pytorch` transformations
#     - The problem is that this doesn't grow the size of the dataset
#     - Instead, it calls randomly the transformation for each image, at each epoch
#     - So at each epoch we have the same number of images, but sometimes transformed an sometimes not
#     - This type of data augmentation doesn't solve our problem
# 2. Use `albumentation` library
#     - Same would happen, as can be seen in the [official docs](https://albumentations.ai/docs/examples/pytorch_classification/)
# 3. Perform data augmentation manually
#     - As we couldn't find a ready-to-use solution, this seems the way to go
#     - Make ourselves the code to perform the data augmentation
#
# So, the **process** is going to be:
#
# 1. Iterate over all classes of the dataset
# 2. If that class has less than `K` images, perform data augmentation to get at least that number of images
# 3. Wrap it on a `Dataset` class for ease of use

## Augmentation of the dataset
# ==============================================================================


# Use the cached dataset
if USE_CACHED_AUGMENTED_DATASET == True:
    train_dataset_augmented = torch.load(AUGMENTED_DATASET_CACHE_FILE)

# We have to do the data augmentation if we mark that we want to do it (section parameter)
# Or if the cached dataset was done for other number of images (ie. for 4 when
# now we want 32)
if USE_CACHED_AUGMENTED_DATASET == False or train_dataset_augmented.min_number_of_images != K:

    # Select the data augmentation mechanism
    AugmentationClass = LazyAugmentatedDataset if LAZY_DATA_AUGMENTATION is True else AugmentatedDataset

    train_dataset_augmented = AugmentationClass(
        base_dataset = train_dataset,
        min_number_of_images = K,

        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(250, 250)),
            transforms.RandomRotation(degrees=(0, 45)),
            transforms.RandomAutocontrast(),
        ])

    )

    # Save the augmented dataset to cache
    torch.save(train_dataset_augmented, AUGMENTED_DATASET_CACHE_FILE)

# Now put a loader in front of the augmented dataset
train_loader_augmented = torch.utils.data.DataLoader(
    train_dataset_augmented,
    batch_size = ONLINE_BATCH_SIZE,
    num_workers = NUM_WORKERS,
    pin_memory = True,
    sampler = CustomSampler(P, K, train_dataset_augmented)
)


## Remove previous datasets
# ==============================================================================
#
# - If we're not doing hyperparameter tuning, we don't need to hold previous dataset and dataloader


if SKIP_HYPERPARAMTER_TUNING is True:
    # We are not using the old dataset and dataloader
    # So delete them to try to have as much RAM as we can
    # Otherwise, train will crash due to lack of RAM
    del train_dataset
    del train_loader

    try_to_clean_memory()


# Choose the loss function to use
# ==============================================================================
#
# - We have so many combinations for loss functions that is not feasible to use one Colab section for each
# - Combinations depend on:
#     1. Batch hard vs Batch all
#     2. Classical triplet loss vs Softplus loss
#     3. All summands mean vs Only > 0 summands mean
# - This election is done in *Global Parameters of the Notebook* section


batch_loss_function = None
if BATCH_TRIPLET_LOSS_FUNCTION == "hard":
    batch_loss_function = BatchHardTripletLoss(MARGIN, use_softplus = USE_SOFTPLUS_LOSS, use_gt_than_zero_mean = USE_GT_ZERO_MEAN_LOSS)
if BATCH_TRIPLET_LOSS_FUNCTION == "all":
    batch_loss_function = BatchAllTripletLoss(MARGIN, use_softplus = USE_SOFTPLUS_LOSS, use_gt_than_zero_mean =  USE_GT_ZERO_MEAN_LOSS)

# Sanity check
if batch_loss_function is None:
    raise Exception(f"BATCH_TRIPLET_LOSS global parameter got unexpected value: {BATCH_TRIPLET_LOSS_FUNCTION}")


# Choose wheter to add embedding norm or not
# ==============================================================================


if ADD_NORM_PENALTY:
    batch_loss_function = AddSmallEmbeddingPenalization(
        base_loss = batch_loss_function,
        penalty_factor = PENALTY_FACTOR,
    )


# Hyperparameter tuning
# ==============================================================================


# TODO -- explore more hyperparameters now that we have compute power
# Controlamos si queremos realizar el hyperparameater tuning o no
if SKIP_HYPERPARAMTER_TUNING is False:

    # The loader used for converting each fold to a DataLoader
    def loader_generator(fold_dataset: split_dataset.WrappedSubset) -> DataLoader:

        loader = torch.utils.data.DataLoader(
            fold_dataset,
            batch_size = ONLINE_BATCH_SIZE,
            shuffle = True,
            num_workers = NUM_WORKERS,
            pin_memory = True,
        )

        return loader

    # The function that takes a training fold loader and a network, and returns
    # a trained net
    def network_trainer(fold_dataloader: DataLoader, net: torch.nn.Module) -> torch.nn.Module:
        _ = train_model_online(
            net = net,
            path = os.path.join(BASE_PATH, "tmp"),
            parameters = parameters,
            train_loader = train_loader,
            validation_loader = validation_loader,
            name = "SiameseNetworkOnline",
            logger = SilentLogger(),
            snapshot_iterations = None,
            gradient_clipping = GRADIENT_CLIPPING
        )

        return net

    # The function that takes a trained net, and the loader for the validation
    # fold, and produces the loss value that we want to optimize
    def loss_function(net: torch.nn.Module, validation_fold: DataLoader) -> float:
        return metrics.silhouette(validation_fold, net)

    # Para controlar que parametros ya hemos explorado y queremos saltar
    already_explored_parameters = [
    ]

    # Parameters that we are going to explore
    margin_values = [0.1, 0.5, 1.0]
    learning_rate_values = [0.00001, 0.0001, 0.001, 0.01]
    embedding_dimension_values = [2, 3, 4, 5, 10, 15]
    network_models = [LFWResNet18, LFWLightModel, LFWResNet18]

    # Parametros que fijamos de antemano
    epochs = HYPERPARAMETER_TUNING_EPOCHS

    # Llevamos la cuenta de los mejores parametros y el mejor error encontrados hasta
    # el momento
    best_loss = None
    best_parameters = {
        "embedding_dimension": None,
        "lr": None,
        "margin": None,
        "model": None
    }

    # Aux function so we decrease the indentation level of the CV for loop
    # TODO -- consider PK values and other important parameters
    def cv_parameters_generator(
        margin_values,
        learning_rate_values,
        embedding_dimension_values,
        network_models
    ):
        for margin in margin_values:
            for learning_rate in learning_rate_values:
                for embedding_dimension in embedding_dimension_values:
                    for network in network_models:
                        yield margin, learning_rate, embedding_dimension, network


    # Exploramos las combinaciones de parametros
    for margin, learning_rate, embedding_dimension, network in cv_parameters_generator(
        margin_values,
        learning_rate_values,
        embedding_dimension_values,
        network_models
    ):

        print(f"Optimizando para margin: {margin}, lr: {learning_rate}, embedding_dim: {embedding_dimension}, model: {network}")

        # Comprobamos si tenemos que saltarnos el calculo de algun valor
        # porque ya se haya hecho
        if (embedding_dimension, learning_rate, margin, network) in already_explored_parameters:
            print("\tSaltando este calculo porque ya se realizo")
            continue

        # Network generator that we use in our custom cross validation
        def network_creator():
            net = network(embedding_dimension)
            net.set_permute(False)
            return net

        parameters = dict()
        parameters["epochs"] = epochs
        parameters["lr"] = learning_rate
        parameters["criterion"] = BatchHardTripletLoss(margin)
        logger = SilentLogger()

        # Run k-fold cross validation for this configuration of parameters
        # For some combinations of parameters, this can fail
        try:
            losses = hptuning.custom_cross_validation(
                train_dataset = train_dataset,
                k = NUMBER_OF_FOLDS,
                random_seed = RANDOM_SEED,
                network_creator = network_creator,
                network_trainer = network_trainer,
                loader_generator = loader_generator,
                loss_function = loss_function,
            )
            print(f"Obtained loss is {losses.mean()}")
            print("")

        except Exception as e:

            # Show that cross validation failed for this combination
            msg = "Could not run succesfully k-fold cross validation for this combination of parameters"
            msg = msg + f"\nError was: {e}"
            print(msg)
            file_logger.warn(msg)

            # Use None loss, that is treated later
            losses = None


        # Check if we have a better loss value
        # First we check that we have a valid loss list
        # Second we check that we have a better value
        valid_loss_values = losses is not None and math.isnan(losses.mean()) is False
        better_loss_value = best_loss is None or losses.mean() < best_loss
        compound_condition = valid_loss_values and better_loss_value
        if compound_condition:

            # Update the best loss value
            best_loss = losses.mean()
            best_parameters = {
                "embedding_dimension": embedding_dimension,
                "lr": learning_rate,
                "margin": margin,
                "network_model": network
            }

            # Show the better values that we have found
            print("==> 🔎 FOUND NEW BEST PARAMETERS")
            print(f"Current best parameters: {best_parameters}")
            print(f"Current best loss: {best_loss}")




# Training of the model
# ==============================================================================

## Selecting the network and tweaking some parameters
# ==============================================================================


net = None
if NET_MODEL == "ResNet18":
    net = ResNet18(EMBEDDING_DIMENSION)
elif NET_MODEL == "LightModel":
    net = LightModel(EMBEDDING_DIMENSION)
elif NET_MODEL == "LFWResNet18":
    net = LFWResNet18(EMBEDDING_DIMENSION)
elif NET_MODEL == "LFWLightModel":
    net = LFWLightModel(EMBEDDING_DIMENSION)
else:
    raise Exception("Parameter 'NET_MODEL' has not a valid value")

# Wrap the model if we want to normalize the output
if NORMALIZED_MODEL_OUTPUT is True:
    net = NormalizedNet(net)

# The custom sampler takes care of minibatch management
# Thus, we don't have to make manipulations on them
net.set_permute(False)

# Training parameters
parameters = dict()
parameters["epochs"] = TRAINING_EPOCHS
parameters["lr"] = ONLINE_LEARNING_RATE

# We use the loss function that depends on the global parameter BATCH_TRIPLET_LOSS_FUNCTION
# We selected this loss func in *Choose the loss function to use* section
parameters["criterion"] = batch_loss_function

print(net)


## Defining the loggers we want to use
# ==============================================================================


# Define the loggers we want to use
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

# Combine them in a single logger
logger = CompoundLogger([
    triplet_loss_logger,
    cluster_sizes_logger,
    intercluster_metrics_logger
])


## Running the training loop
# ==============================================================================


import torch

# Check if we want to skip training
if USE_CACHED_MODEL is False:

    # To measure the time it takes to train
    ts = time.time()

    # Run the training with or without profiling
    if PROFILE_TRAINING is True:
        training_history = cProfile.run(
            f"""train_model_online(
                net = net,
                path = os.path.join(BASE_PATH, 'tmp'),
                parameters = parameters,
                train_loader = train_loader_augmented,
                validation_loader = validation_loader,
                name = NET_MODEL,
                logger = logger,
                snapshot_iterations = None,
                gradient_clipping = GRADIENT_CLIPPING
            )""",
            PROFILE_SAVE_FILE
        )

    else:

        training_history = train_model_online(
            net = net,
            path = os.path.join(BASE_PATH, "tmp"),
            parameters = parameters,
            train_loader = train_loader_augmented,
            validation_loader = validation_loader,
            name = NET_MODEL,
            logger = logger,
            snapshot_iterations = None,
            gradient_clipping = GRADIENT_CLIPPING
        )

    # Compute how long it took
    te = time.time()
    print(f"It took {te - ts} seconds to train")

    # Update the model cache
    filesystem.save_model(net, MODEL_CACHE_FOLDER, "online_model_cached")

# In case we skipped training, load the model from cache
else:

    # Load the model from cache
    net = filesystem.load_model(
        os.path.join(MODEL_CACHE_FOLDER, "online_model_cached"),
        lambda: LFWResNet18(EMBEDDING_DIMENSION)
    )

    # Load the network in corresponding mem device (cpu -> ram, gpu -> gpu mem
    device = core.get_device()
    net.to(device)



# From this point, we won't perform training on the model
# So eval mode is set for better performance
net.eval()


# Model evaluation
# ==============================================================================

# We start computing the *silhouette* metric for the produced embedding, on
# train, validation and test set:


with torch.no_grad():

    # Try to clean memory, because we can easily run out of memory
    # This provoke the notebook to crash, and all in-memory objects to be lost
    try_to_clean_memory()

    train_silh = metrics.silhouette(train_loader_augmented, net)
    print(f"Silhouette in training loader: {train_silh}")

    validation_silh = metrics.silhouette(validation_loader, net)
    print(f"Silhouette in validation loader: {validation_silh}")

    test_silh = metrics.silhouette(test_loader, net)
    print(f"Silhouette in test loader: {test_silh}")

    # Put this info in wandb
    wandb.log({
        "Training silh": train_silh,
        "Validation silh": validation_silh,
        "Test silh": test_silh
    })


# Show the "criterion" metric on test set


with torch.no_grad():
    net.set_permute(False)

    core.test_model_online(net, test_loader, parameters["criterion"], online = True)

    net.set_permute(True)


# Now take the classifier from the embedding and use it to compute some classification metrics:


with torch.no_grad():

    # Try to clean memory, because we can easily run out of memory
    # This provoke the notebook to crash, and all in-memory objects to be lost
    try_to_clean_memory()

    # With hopefully enough memory, try to convert the embedding to a classificator
    classifier = EmbeddingToClassifier(net, k = NUMBER_NEIGHBOURS, data_loader = train_loader_augmented, embedding_dimension = EMBEDDING_DIMENSION)


# We evaluate this classifier by watching how it works over a small test set. Later we take some metrics from this classifier to evaluate it more precisely.


with torch.no_grad():

    # Shoow only `max_iterations` classifications
    counter = 0
    max_iterations = 20

    for img, img_class in test_dataset:
        predicted_class = classifier.predict(img)
        print(f"True label: {img_class}, predicted label: {predicted_class[0]}, correct: {img_class == predicted_class[0]}")

        counter += 1
        if counter == max_iterations: break


# Plot of the embedding
# ==============================================================================
#
# - If the dimension of the embedding is 2, then we can plot how the transformation to a classificator works:


with torch.no_grad():
    classifier.scatter_plot()


# Evaluating the obtained classifier
# ==============================================================================
#
# - Now that we adapted our network to a classification task, we can compute some classification metrics


with torch.no_grad():
    try_to_clean_memory()
    classifier.embedder.set_permute(False)

    metrics = evaluate_model(classifier, train_loader, test_loader)
    pprint(metrics)

    classifier.embedder.set_permute(True)

