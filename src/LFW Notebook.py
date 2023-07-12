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
#
# - For ease of use, we are going to store all global parameters into a dict
# - This way, we can pass this dict directly to wandb init, so we can keep track
# of which parameters produced which output

from typing import Dict, Union
GLOBALS: Dict[str, Union[str, int, float, bool]] = dict()

## Paths
# ==============================================================================
#
# - Parameters related to data / model / lib paths

# Lib to define paths
import os

# Define if we are running the notebook in our computer ("local")
# or in Google Colab ("remote")
# or in UGR's server ("ugr")
GLOBALS['RUNNING_ENV'] = "ugr"

# Base path for the rest of paths defined in the notebook
GLOBALS['BASE_PATH'] = None
if GLOBALS['RUNNING_ENV'] == "local":
    GLOBALS['BASE_PATH'] = "./"
elif GLOBALS['RUNNING_ENV'] == "remote":
    GLOBALS['BASE_PATH'] = "/content/drive/MyDrive/Colab Notebooks/"
elif GLOBALS['RUNNING_ENV'] == "ugr":
    GLOBALS['BASE_PATH'] = "/mnt/homeGPU/squijano/TFG/"
else:
    raise Exception(f"RUNNING ENV is not valid, got value {GLOBALS['RUNNING_ENV']}")

# Path to our lib dir
GLOBALS['LIB_PATH'] = os.path.join(GLOBALS['BASE_PATH'], "lib")

# Path where we store training / test data
GLOBALS['DATA_PATH'] = os.path.join(GLOBALS['BASE_PATH'], "data")

# Dir with all cached models
# This cached models can be loaded from disk when training is skipped
GLOBALS['MODEL_CACHE_FOLDER'] = os.path.join(GLOBALS['BASE_PATH'], "cached_models")

# Cache for the augmented dataset
GLOBALS['AUGMENTED_DATASET_CACHE_FILE'] = os.path.join(GLOBALS['BASE_PATH'], "cached_augmented_dataset.pt")

# File where the logs are written
GLOBALS['LOGGING_FILE'] = os.path.join(GLOBALS['BASE_PATH'], "training.log")

# Binary file where the stats of the profiling are saved
GLOBALS['PROFILE_SAVE_FILE'] = os.path.join(GLOBALS['BASE_PATH'], "training_profile.stat")

GLOBALS['OPTUNA_DATABASE'] = f"sqlite:///{GLOBALS['BASE_PATH']}/hp_tuning_optuna.db"

## ML parameters
# ==============================================================================
#
# - Parameters related to machine learning
# - For example, batch sizes, learning rates, ...


# Parameters of P-K sampling
GLOBALS['P'] = 34    # Number of classes used in each minibatch
GLOBALS['K'] = 2     # Number of images sampled for each selected class

# Batch size for online training
# We can use `P * K` as batch size. Thus, minibatches will be
# as we expect in P-K sampling.
#
# But we can use `n * P * K`. Thus, minibatches will be n P-K sampling
# minibatche concatenated together
# Be careful when doing this because it can be really slow, and there is no
# clear reason to do this
GLOBALS['ONLINE_BATCH_SIZE'] = GLOBALS['P'] * GLOBALS['K']

# Epochs for hard triplets, online training
GLOBALS['TRAINING_EPOCHS'] = 500

# Learning rate for hard triplets, online training
GLOBALS['ONLINE_LEARNING_RATE'] = 0.00005

# How many single elements we want to see before logging
# It has to be a multiple of P * K, otherwise `should_log` would return always
# false as `it % LOGGING_ITERATIONS != 0` always
#
# `LOGGING_ITERATIONS = P * K * n` means we log after seeing `n` P-K sampled
# minibatches
GLOBALS['LOGGING_ITERATIONS'] = GLOBALS['P'] * GLOBALS['K'] * 100

# Which percentage of the training and validation set we want to use for the logging
GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'] = 1 / 10
GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'] = 1 / 3

# Choose which model we're going to use
# Can be "ResNet18", "LightModel", "LFWResNet18" or "LFWLightModel"
GLOBALS['NET_MODEL'] = "LFWResNet18"

# Epochs used in k-Fold Cross validation
# k-Fold Cross validation used for parameter exploration
GLOBALS['HYPERPARAMETER_TUNING_EPOCHS'] = 20

# Number of tries in the optimization process
# We are using optuna, so we try `HYPERPARAMETER_TUNING_TRIES` times with different
# hyperparameter configurations
GLOBALS['HYPERPARAMETER_TUNING_TRIES'] = 300

# Number of folds used in k-fold Cross Validation
GLOBALS['NUMBER_OF_FOLDS'] = 8

# Margin used in the loss function
GLOBALS['MARGIN'] = 0.5

# Dim of the embedding calculated by the network
GLOBALS['EMBEDDING_DIMENSION'] = 8

# Number of neighbours considered in K-NN
# K-NN used for transforming embedding task to classification task
GLOBALS['NUMBER_NEIGHBOURS'] = 4

# Batch Triplet Loss Function
# This way we can choose among "hard", "all"
GLOBALS['BATCH_TRIPLET_LOSS_FUNCTION'] = "hard"

# Whether or not use softplus loss function instead of vanilla triplet loss
GLOBALS['USE_SOFTPLUS_LOSS'] = True

# Count all sumamnds in the mean loss or only those summands greater than zero
GLOBALS['USE_GT_ZERO_MEAN_LOSS'] = True

# Wether or not use lazy computations in the data augmentation
GLOBALS['LAZY_DATA_AUGMENTATION'] = True

# Where or not add penalty term to the loss function
GLOBALS['ADD_NORM_PENALTY'] = True

# If we add that penalty term, which scaling factor to use
GLOBALS['PENALTY_FACTOR'] = 0.6

# If we want to wrap our model into a normalizer
# That wrapper divides each vector by its norm, thus, forcing norm 1 on each vector
GLOBALS['NORMALIZED_MODEL_OUTPUT'] = True

# If its None, we do not perform gradient clipping
# If its a Float value, we perform gradient clipping, using that value as a
# parameter for the max norm
GLOBALS['GRADIENT_CLIPPING'] = 100


## Section parameters
# ==============================================================================

# - Flags to choose if some sections will run or not
# - This way we can skip some heavy computations when not needed


# Skip hyper parameter tuning for online training
GLOBALS['SKIP_HYPERPARAMTER_TUNING'] = True

# Skip training and use a cached model
# Useful for testing the embedding -> classifier transformation
# Thus, when False training is not computed and a cached model
# is loaded from disk
# Cached models are stored in `MODEL_CACHE_FOLDER`
GLOBALS['USE_CACHED_MODEL'] = False

# Skip data augmentation and use the cached augmented dataset
GLOBALS['USE_CACHED_AUGMENTED_DATASET'] = False

# Most of the time we're not exploring the data, but doing
# either hyperparameter settings or training of the model
# So if we skip this step we can start the process faster
GLOBALS['SKIP_EXPLORATORY_DATA_ANALYSYS'] = True

# Wether or not profile the training
# This should be False most of the times
# Note that profiling adds a significant overhead to the training
GLOBALS['PROFILE_TRAINING'] = False


## WANDB Parameters
# ==============================================================================

from datetime import datetime

# Name for the project
# One project groups different runs
GLOBALS['WANDB_PROJECT_NAME'] = "Labeled Faces in the Wild (LFW) Dataset"

# Name for this concrete run
# I don't care too much about it, because wandb tracks the parameters we use
# in this run (see "Configuration for Weights and Biases" section)
GLOBALS['WANDB_RUN_NAME'] = str(datetime.now())


## Others
# ==============================================================================

# Number of workers we want to use
# We can have less, equal or greater num of workers than CPUs
# In the following forum:
#   https://discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/4
# they recomend to explore this parameter, growing it until system RAM saturates
# Using a value greater than 2 makes pytorch tell us that this value is not optimal
# So sticking with what pytorch tells uss
# TODO -- trying a higher value in UGR's server
GLOBALS['NUM_WORKERS'] = 4

# Fix random seed to make reproducible results
GLOBALS['RANDOM_SEED'] = 123456789

# Add some paths to PYTHONPATH
# ==============================================================================

# Python paths are difficult to manage
# In this script we can do something like:
# `import lib.core as core` and that's fine
# But in lib code we cannot import properly the modules

import sys
sys.path.append(os.path.join(GLOBALS['BASE_PATH'], "src"))
sys.path.append(os.path.join(GLOBALS['BASE_PATH'], "src/lib"))
sys.path.append(GLOBALS['BASE_PATH'])

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

import optuna

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

# Server security check
# ==============================================================================
#
# - Sometimes UGR's server does not provide GPU access
# - In that case, fail fast so we start ASAP debugging the problem

if GLOBALS['RUNNING_ENV'] == "ugr" and torch.cuda.is_available() is False:
    raise Exception("`torch.cuda.is_available()` returned false, so we dont have access to GPU's")


# Configuration of the logger
# ==============================================================================
#
# - Here we set the configuration for all logging done
# - In lib, `logging.getLogger("MAIN_LOGGER")` is used everywhere, so we get it, configure it once, and use that config everywhere


# Get the logger that is used everywhere
file_logger = logging.getLogger("MAIN_LOGGER")

# Configure it
# Avoid propagating to upper logger, which logs to the console
file_logger.propagate = False

file_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s::%(levelname)s::%(funcName)s::> %(message)s")
file_handler = logging.FileHandler(GLOBALS['LOGGING_FILE'])
file_handler.setFormatter(formatter)
file_logger.addHandler(file_handler)

# 'application' code
file_logger.debug('debug message')


# Configuration for Weigths and Biases
# ==============================================================================
#
# - We're going to use `wandb` for tracking the training of the models
# - We use our `GLOBALS` dict to init wandb, that is going to keep track of all
# of that parameters

# If we're running in UGR's servers, we need to set some ENV vars
# Otherwise, wandb is going to write to dirs that it has no access
# Also, pytorch tries to save pretrained models in the home folder
if GLOBALS['RUNNING_ENV'] == "ugr":

    print("-> Changing dir env values")
    utils.change_dir_env_vars(base_path = GLOBALS['BASE_PATH'])
    print("-> Changing done!")
    print("")

    print("-> Login again to WANDB")
    utils.login_wandb()
    print("-> Login done!")
    print("")

# Init the wandb tracker
# We need to do this before `wandb.login`
wandb.init(
    project = GLOBALS['WANDB_PROJECT_NAME'],
    name = GLOBALS['WANDB_RUN_NAME'],
    config = GLOBALS,
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
    transforms.Resize((250, 250), antialias=True),
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
    root = GLOBALS['DATA_PATH'],
    split = "train",
    download = True,
    transform = transform,
)

test_dataset = torchvision.datasets.LFWPeople(
    root = GLOBALS['DATA_PATH'],
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
    batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
    num_workers = GLOBALS['NUM_WORKERS'],
    pin_memory = True,
    sampler = CustomSampler(GLOBALS['P'], GLOBALS['K'], train_dataset)
)

# TODO -- here I don't know if use default sampler or custom sampler
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
    shuffle = True,
    num_workers = GLOBALS['NUM_WORKERS'],
    pin_memory = True,
)

# TODO -- here I don't know if use default sampler or custom sampler
test_loader = torch.utils.data.DataLoader(
  test_dataset,
  batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
  shuffle = True,
  num_workers = GLOBALS['NUM_WORKERS'],
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
if GLOBALS['USE_CACHED_AUGMENTED_DATASET'] == True:
    train_dataset_augmented = torch.load(GLOBALS['AUGMENTED_DATASET_CACHE_FILE'])

# We have to do the data augmentation if we mark that we want to do it (section parameter)
# Or if the cached dataset was done for other number of images (ie. for 4 when
# now we want 32)
if GLOBALS['USE_CACHED_AUGMENTED_DATASET'] == False or train_dataset_augmented.min_number_of_images != GLOBALS['K']:

    # Select the data augmentation mechanism
    AugmentationClass = LazyAugmentatedDataset if GLOBALS['LAZY_DATA_AUGMENTATION'] is True else AugmentatedDataset

    train_dataset_augmented = AugmentationClass(
        base_dataset = train_dataset,
        min_number_of_images = GLOBALS['K'],

        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(250, 250), antialias = True),
            transforms.RandomRotation(degrees=(0, 45)),
            transforms.RandomAutocontrast(),
        ])

    )

    # Save the augmented dataset to cache
    torch.save(train_dataset_augmented, GLOBALS['AUGMENTED_DATASET_CACHE_FILE'])

# Now put a loader in front of the augmented dataset
train_loader_augmented = torch.utils.data.DataLoader(
    train_dataset_augmented,
    batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
    num_workers = GLOBALS['NUM_WORKERS'],
    pin_memory = True,
    sampler = CustomSampler(GLOBALS['P'], GLOBALS['K'], train_dataset_augmented)
)


## Remove previous datasets
# ==============================================================================
#
# - If we're not doing hyperparameter tuning, we don't need to hold previous dataset and dataloader


if GLOBALS['SKIP_HYPERPARAMTER_TUNING'] is True:
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
if GLOBALS['BATCH_TRIPLET_LOSS_FUNCTION'] == "hard":
    batch_loss_function = BatchHardTripletLoss(GLOBALS['MARGIN'], use_softplus = GLOBALS['USE_SOFTPLUS_LOSS'], use_gt_than_zero_mean = GLOBALS['USE_GT_ZERO_MEAN_LOSS'])
if GLOBALS['BATCH_TRIPLET_LOSS_FUNCTION'] == "all":
    batch_loss_function = BatchAllTripletLoss(GLOBALS['MARGIN'], use_softplus = GLOBALS['USE_SOFTPLUS_LOSS'], use_gt_than_zero_mean =  GLOBALS['USE_GT_ZERO_MEAN_LOSS'])

# Sanity check
if batch_loss_function is None:
    raise Exception(f"BATCH_TRIPLET_LOSS global parameter got unexpected value: {GLOBALS['BATCH_TRIPLET_LOSS_FUNCTION']}")


# Choose wheter to add embedding norm or not
# ==============================================================================


if GLOBALS['ADD_NORM_PENALTY']:
    batch_loss_function = AddSmallEmbeddingPenalization(
        base_loss = batch_loss_function,
        penalty_factor = GLOBALS['PENALTY_FACTOR'],
    )


# Hyperparameter tuning
# ==============================================================================

# The following function is a parameters for our `custom_cross_validation`
# It does not depend on the hyperparameters that we are exploring, so we can
# define it here
#
# The function that takes a trained net, and the loader for the validation
# fold, and produces the loss value that we want to optimize
def loss_function(net: torch.nn.Module, validation_fold: DataLoader) -> float:
    return metrics.silhouette(validation_fold, net)

def objective(trial):
    """Optuna function that is going to be used in the optimization process"""

    # Fixed parameters
    # This parameters were explored using previous hp tuning experiments
    # Fixing the parameters lets us explore other parameters in a better way
    learning_rate = 0.00005
    net_election = "LFWResNet18"
    softplus = True
    margin = 0.5
    use_norm_penalty = True
    norm_penalty = 0.6
    normalization_election = True
    p = 34
    k = 2
    embedding_dimension = 8

    # Log that we are going to do k-fold cross validation and the values of the
    # parameters. k-fold cross validation can be really slow, so this logs are
    # useful for babysitting the process
    print("")
    print(f"🔎 Starting cross validation for trial {trial.number}")
    print(f"🔎 Parameters for this trial are:\n{trial.params}")
    print("")

    # With all parameters set, we can create all the necessary elements for
    # running k-fold cross validation with this configuration

    # With P, K values, we can generate the augmented dataset
    train_dataset_augmented = LazyAugmentatedDataset(
        base_dataset = train_dataset,
        min_number_of_images = k,

        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(250, 250), antialias = True),
            transforms.RandomRotation(degrees=(0, 45)),
            transforms.RandomAutocontrast(),
        ])

    )

    # And with p, k values we can define the way we use the laoder generator
    # This p, k values are captured in the outer scope for the `CustomSampler`
    def loader_generator(
        fold_dataset: split_dataset.WrappedSubset,
        fold_type: hptuning.FoldType
    ) -> DataLoader:


        # When doing the split, we can end with less than p classes with at least
        # k images associated. So we do an augmentation again:
        #
        # This assures that all classes have at least k images, so they are not
        # erased in `LazyAugmentatedDataset`. But this does not create more
        # classes, if fold has already less than P classes
        #
        # We only perform data augmentation in the training fold
        #
        # TODO -- this can affect the optimization process, but otherwise most
        # of the tries will fail because of this problem
        if fold_type is hptuning.FoldType.TRAIN_FOLD:
            fold_dataset_augmented = LazyAugmentatedDataset(
                base_dataset = fold_dataset,
                min_number_of_images = k,

                # Remember that the trasformation has to be random type
                # Otherwise, we could end with a lot of repeated images
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=(250, 250), antialias = True),
                    transforms.RandomRotation(degrees=(0, 45)),
                    transforms.RandomAutocontrast(),
                ])

            )
        else:
            # Do not perform any augmentation
            fold_dataset_augmented = fold_dataset

        # In the same line, we only use our custom sampler for the training fold
        # Otherwise, we use a normal sampler
        if fold_type is hptuning.FoldType.TRAIN_FOLD:
            loader = torch.utils.data.DataLoader(
                fold_dataset_augmented,
                batch_size = p * k,
                num_workers = GLOBALS['NUM_WORKERS'],
                pin_memory = True,
                sampler = CustomSampler(p, k, fold_dataset)
            )
        elif fold_type is hptuning.FoldType.VALIDATION_FOLD:
            loader = torch.utils.data.DataLoader(
                fold_dataset_augmented,
                batch_size = p * k,
                num_workers = GLOBALS['NUM_WORKERS'],
                pin_memory = True,
            )
        else:
            raise ValueError(f"{fold_type} enum value is not managed in if elif construct!")

        # To avoid accessing loader.__len__ without computing the necessary data
        _ = loader.__iter__()

        return loader

    # Wrap the network in a lambda function so we can use it in `custom_cross_validation`
    def network_creator():

        # Model that we have chosen
        if net_election == "LFWResNet18":
            net = LFWResNet18(embedding_dimension = embedding_dimension)
        elif net_election == "LFWLightModel":
            net = LFWLightModel(embedding_dimension = embedding_dimension)
        else:
            raise ValueError("String for net election is not valid")

        # Wether or not use normalization
        if normalization_election is True:
            net = NormalizedNet(net)

        net.set_permute(False)
        return net

    # The function that takes a training fold loader and a network, and returns
    # a trained net. This is a parameter for our `custom_cross_validation`
    def network_trainer(fold_dataloader: DataLoader, net: torch.nn.Module) -> torch.nn.Module:

        parameters = dict()
        parameters["epochs"] = GLOBALS['HYPERPARAMETER_TUNING_EPOCHS']
        parameters["lr"] = learning_rate
        parameters["criterion"] = BatchHardTripletLoss(margin, use_softplus = softplus)

        # Wether or not use norm penalization
        if use_norm_penalty:
            parameters["criterion"] = AddSmallEmbeddingPenalization(
                base_loss = parameters["criterion"],
                penalty_factor = norm_penalty,
            )

        _ = train_model_online(
            net = net,
            path = os.path.join(GLOBALS['BASE_PATH'], "tmp_hp_tuning"),
            parameters = parameters,
            train_loader = fold_dataloader,
            validation_loader = None,
            name = "Hyperparameter Tuning Network",
            logger = SilentLogger(),
            snapshot_iterations = None,
            gradient_clipping = GLOBALS['GRADIENT_CLIPPING'],
            fail_fast = True,
        )

        return net


    # Now we have defined everything for `custom_cross_validation`. So we can
    # run k-fold cross validation for this configuration of parameters
    # For some combinations of parameters, this can fail
    try:
        losses = hptuning.custom_cross_validation(
            train_dataset = train_dataset_augmented,
            k = GLOBALS['NUMBER_OF_FOLDS'],
            random_seed = GLOBALS['RANDOM_SEED'],
            network_creator = network_creator,
            network_trainer = network_trainer,
            loader_generator = loader_generator,
            loss_function = loss_function,
        )
        print(f"Obtained loss (cross validation mean) is {losses.mean()}")
        print("")

    except Exception as e:

        # Show that cross validation failed for this combination
        msg = "Could not run succesfully k-fold cross validation for this combination of parameters"
        msg = msg + f"\nError was: {e}"
        print(msg)
        file_logger.warn(msg)

        # Return None so this trial is not considered
        return None


    # If everything went alright, return the mean of the loss
    return losses.mean()


if GLOBALS['SKIP_HYPERPARAMTER_TUNING'] is False:

    # We want to maximize silhouete value
    print("🔎 Started hyperparameter tuning")

    study = optuna.create_study(
        direction = "maximize",
        study_name = "Silhouette optimization",
        storage = GLOBALS['OPTUNA_DATABASE'],
        load_if_exists = True
    )
    study.optimize(objective, n_trials = GLOBALS['HYPERPARAMETER_TUNING_TRIES'])

    print("🔎 Hyperparameter tuning ended")
    print("")
    print(f"🔎 Best trial: {study.best_trial}")
    print("")


# Training of the model
# ==============================================================================

## Selecting the network and tweaking some parameters
# ==============================================================================


net = None
if GLOBALS['NET_MODEL'] == "ResNet18":
    net = ResNet18(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "LightModel":
    net = LightModel(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "LFWResNet18":
    net = LFWResNet18(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "LFWLightModel":
    net = LFWLightModel(GLOBALS['EMBEDDING_DIMENSION'])
else:
    raise Exception("Parameter 'NET_MODEL' has not a valid value")

# Wrap the model if we want to normalize the output
if GLOBALS['NORMALIZED_MODEL_OUTPUT'] is True:
    net = NormalizedNet(net)

# The custom sampler takes care of minibatch management
# Thus, we don't have to make manipulations on them
net.set_permute(False)

# Training parameters
parameters = dict()
parameters["epochs"] = GLOBALS['TRAINING_EPOCHS']
parameters["lr"] = GLOBALS['ONLINE_LEARNING_RATE']

# We use the loss function that depends on the global parameter BATCH_TRIPLET_LOSS_FUNCTION
# We selected this loss func in *Choose the loss function to use* section
parameters["criterion"] = batch_loss_function

print(net)


## Defining the loggers we want to use
# ==============================================================================


# Define the loggers we want to use
triplet_loss_logger = TripletLoggerOnline(
    net = net,
    iterations = GLOBALS['LOGGING_ITERATIONS'],
    loss_func = parameters["criterion"],
    train_percentage = GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'],
    validation_percentage = GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'],
    greater_than_zero = GLOBALS['USE_GT_ZERO_MEAN_LOSS'],
)

cluster_sizes_logger = IntraClusterLogger(
    net = net,
    iterations = GLOBALS['LOGGING_ITERATIONS'],
    train_percentage = GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'],
    validation_percentage = GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'],
)

intercluster_metrics_logger = InterClusterLogger(
    net = net,
    iterations = GLOBALS['LOGGING_ITERATIONS'],
    train_percentage = GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'],
    validation_percentage = GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'],
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
if GLOBALS['USE_CACHED_MODEL'] is False:

    # To measure the time it takes to train
    ts = time.time()

    # Run the training with or without profiling
    if GLOBALS['PROFILE_TRAINING'] is True:
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
            GLOBALS['PROFILE_SAVE_FILE']
        )

    else:

        training_history = train_model_online(
            net = net,
            path = os.path.join(GLOBALS['BASE_PATH'], "tmp"),
            parameters = parameters,
            train_loader = train_loader_augmented,
            validation_loader = validation_loader,
            name = GLOBALS['NET_MODEL'],
            logger = logger,
            snapshot_iterations = None,
            gradient_clipping = GLOBALS['GRADIENT_CLIPPING']
        )

    # Compute how long it took
    te = time.time()
    print(f"It took {te - ts} seconds to train")

    # Update the model cache
    filesystem.save_model(net, GLOBALS['MODEL_CACHE_FOLDER'], "online_model_cached")

# In case we skipped training, load the model from cache
else:

    # Load the model from cache
    net = filesystem.load_model(
        os.path.join(GLOBALS['MODEL_CACHE_FOLDER'], "online_model_cached"),

        # TODO -- BUG -- we are not taking in consideration `GLOBALS['NET_MODEL']`
        lambda: LFWResNet18(GLOBALS['EMBEDDING_DIMENSION'])
    )

    # Load the network in corresponding mem device (cpu -> ram, gpu -> gpu mem
    device = core.get_device()
    net.to(device)



# From this point, we won't perform training on the model
# So eval mode is set for better performance
net.eval()


# Model evaluation
# ==============================================================================

# TODO -- testing out ideas
# TODO -- clean and move to propper place in the codebase
# |>

class RetrievalAdapter(torch.nn.Module):
    """
    Takes a network that computes embeddings of the input images and adapts it
    to a network that does a retrieval task

    That is to say, we end up with a network that, given a query image and a set
    of candidates, returns a ranked list of the most promising candidates
    """

    def __init__(self, base_net: torch.nn.Module):
        super(RetrievalAdapter, self).__init__()
        self.base_net = base_net

        # Put the base model on the proper device
        device = core.get_device()
        self.base_net.to(device)

    def query(self, query: torch.Tensor, candidates: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Given a `query` image, and a list of image `candidates` (list in the form
        of a pytorch tensor), returns the `k` most promising candidates, ranked
        by relevance (that is to say, the first element of the list should be more
        similar to the `query` than the last element of the list)

        NOTE: Even though `query` is a single image, it has to be a `torch.Tensor`
        with shape `[1, channels, witdh, height]`, that is to say, it has to be a
        batch of a single element. This is the shape that most of the networks
        we are working with accept
        """

        print(f"TODO -- {self.base_net=}")
        print("")

        # Check the dimensions of the query
        # Query is a single image so `query.shape == [1, 1 | 3, width, height]`
        # Note that it could be `query.shape == [1 | 3, width, height]`, but to be able
        # to pass the image through the network, we need to have a batch of
        # a single image
        if len(query.shape) != 4:
            raise ValueError(f"`query` should be a tensor with four modes, only {len(query.shape)} modes were found")

        if query.shape[0] != 1:
            raise ValueError(f"`query` must be a tensor of batch size 1, batch size found was {query.shape[0]}\nTODO -- {query.shape=}")

        if query.shape[1] != 3 and query.shape[1] != 1:
            raise ValueError(f"`query` must be an image with one or three channels, got {query.shape[1]} channels \nTODO -- {query.shape=}")

        # Check the dimensions of the candidates
        # `candidates` is a list of images, so `candidates.shape == [n, channels = 1 | 3, width, height]`
        # Also, as we are querying for the best `k` candidates, we should have at least
        # `k` candidates
        if len(candidates.shape) != 4:
            raise ValueError(f"`candidates` should be a tensor with four modes, only {len(candidates.shape)} modes were found")

        if candidates.shape[1] != 3 and candidates.shape[1] != 1:
            raise ValueError(f"Candidates must be images of one or three channels, got {candidates.shape[1]} channels")

        if candidates.shape[0] < k:
            raise ValueError(f"Querying for the best {k} candidates, but we only have {candidates.shape[0]} candidates in total\nTODO -- {candidates.shape=}")

        # Make sure that both query and candidates tensors are in the proper device
        # Also, check that the network is in the proper device
        device = core.get_device()
        self.base_net.to(device) # TODO -- remove, should be useless
        query.to(device)
        candidates.to(device)

        # Compute the embeddings of the images
        query_embedding = self.base_net(query)
        print("TODO -- query embedding computed")
        candidate_embeddings = self.base_net(candidates)
        print("TODO -- candidate embeddings computed!")

        pass

    def set_permute(self, should_permute: bool):
        self.base_net.set_permute(should_permute)


# TODO -- remove this monkey patching
core.get_device = lambda: "cpu"

# Wrap the net to perform retrieval
retrieval_net = RetrievalAdapter(net)

# Test a single query to check that all is working
# We are going to use our custom sampler, this way we have P-K sampling. Therefore
# we have P classes with K images each one. If the network were perfect, a
# `k-1` query should return only images of the same class

# Get the first batch of the training set using our custom sampler
example_batch = None
for batch in train_loader_augmented:
    example_batch = batch
    break
example_imgs, example_labels = example_batch

query_img, query_label = example_imgs[0], example_labels[0]
query_img = torch.unsqueeze(query_img, 0)  # We want a batch of a single element
print(f"Batched query now looks like: {query_img=}")
print(f"Batched query now has shape: {query_img.shape=}")
candidate_imgs, candidate_labels = example_imgs[1:], example_labels[1:]

result = retrieval_net.query(query_img, candidate_imgs, GLOBALS['K'] - 1)
raise ValueError("All went good! :D")

# TODO -- end of testing of ideas
# <|
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
    classifier = EmbeddingToClassifier(net, k = GLOBALS['NUMBER_NEIGHBOURS'], data_loader = train_loader_augmented, embedding_dimension = GLOBALS['EMBEDDING_DIMENSION'])


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

