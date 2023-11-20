# FG-NET
# ==============================================================================
#
# - Check `FG-Net Notebook.ipynb` for:
    #  - Some notes on other papers related to our work
    #  - EDA of the dataset

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
GLOBALS['CACD_DATA_PATH'] = os.path.join(GLOBALS['BASE_PATH'], "data/CACD")

# Dataset has images and metadata. Here we store the path to the img dir
GLOBALS['IMAGE_DIR_PATH'] = os.path.join(GLOBALS['DATA_PATH'], "FGNET/images")
GLOBALS['CACD_IMAGE_DIR_PATH'] = os.path.join(GLOBALS['CACD_DATA_PATH'], "CACD2000")

# URL where the datasets are stored
GLOBALS['DATASET_URL'] = "http://yanweifu.github.io/FG_NET_data/FGNET.zip"
GLOBALS['CACD_DATASET_URL'] = "https://drive.google.com/file/d/1hYIZadxcPG27Fo7mQln0Ey7uqw1DoBvM/view"

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
GLOBALS['P'] = 8     # Number of classes used in each minibatch
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

# Training epochs
GLOBALS["TRAINING_EPOCHS"] = 1

# Learning rate for hard triplets, online training
GLOBALS['ONLINE_LEARNING_RATE'] = 5.157 * 10**(-4)

# How many single elements we want to see before logging
# It has to be a multiple of P * K, otherwise `should_log` would return always
# false as `it % LOGGING_ITERATIONS != 0` always
#
# `LOGGING_ITERATIONS = P * K * n` means we log after seeing `n` P-K sampled
# minibatches
#  GLOBALS['LOGGING_ITERATIONS'] = GLOBALS['P'] * GLOBALS['K'] * 500
GLOBALS['LOGGING_ITERATIONS'] = GLOBALS['P'] * GLOBALS['K'] * 1_000

# Which percentage of the training and validation set we want to use for the logging
GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'] = 0.005
GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'] = 0.005

# Choose which model we're going to use
# Can be "ResNet18", "LightModel", "LFWResNet18", "LFWLightModel", "FGLightModel",
#        "CACDResNet18", "CACDResNet50"
GLOBALS['NET_MODEL'] = "CACDResNet18"

# Epochs used in k-Fold Cross validation
# k-Fold Cross validation used for parameter exploration
# TODO -- delete this, we are going to perform a search in the number of epochs
GLOBALS['HYPERPARAMETER_TUNING_EPOCHS'] = 1

# Number of tries in the optimization process
# We are using optuna, so we try `HYPERPARAMETER_TUNING_TRIES` times with different
# hyperparameter configurations
GLOBALS['HYPERPARAMETER_TUNING_TRIES'] = 300

# Wether to use the validation set in the hp tuning process or to use k-fold
# cross validation (which is more robust but way slower)
GLOBALS['FAST_HP_TUNING'] = True

# Number of folds used in k-fold Cross Validation
GLOBALS['NUMBER_OF_FOLDS'] = 2

# Margin used in the loss function
GLOBALS['MARGIN'] = 0.840

# Dim of the embedding calculated by the network
GLOBALS['EMBEDDING_DIMENSION'] = 9

# Number of neighbours considered in K-NN
# K-NN used for transforming embedding task to classification task
GLOBALS['NUMBER_NEIGHBOURS'] = 4

# Batch Triplet Loss Function
# This way we can choose among "hard", "all"
GLOBALS['BATCH_TRIPLET_LOSS_FUNCTION'] = "hard"

# Whether or not use softplus loss function instead of vanilla triplet loss
GLOBALS['USE_SOFTPLUS_LOSS'] = False

# Count all sumamnds in the mean loss or only those summands greater than zero
GLOBALS['USE_GT_ZERO_MEAN_LOSS'] = True

# Wether or not use lazy computations in the data augmentation
GLOBALS['LAZY_DATA_AUGMENTATION'] = True

# Wether or not fail when calling `CustomSampler.__len__` without having previously
# computed the index list
GLOBALS['AVOID_CUSTOM_SAMPLER_FAIL'] = True

# Where or not add penalty term to the loss function
GLOBALS['ADD_NORM_PENALTY'] = False

# If we add that penalty term, which scaling factor to use
GLOBALS['PENALTY_FACTOR'] = 0.6

# If we want to wrap our model into a normalizer
# That wrapper divides each vector by its norm, thus, forcing norm 1 on each vector
GLOBALS['NORMALIZED_MODEL_OUTPUT'] = False

# If its None, we do not perform gradient clipping
# If its a Float value, we perform gradient clipping, using that value as a
# parameter for the max norm
GLOBALS['GRADIENT_CLIPPING'] = None

# Number of candidates that we are going to consider in the retrieval task,
# used in the Rank@K accuracy metric
# We use k = 1 and k = this value
GLOBALS['ACCURACY_AT_K_VALUE'] = 5

# Images in this dataset have different shapes. So this parameter fixes one shape
# so we can normalize the images to have the same shape
GLOBALS['IMAGE_SHAPE'] = (200, 200)

# Degrees that we are going to use in data augmentation rotations
GLOBALS['ROTATE_AUGM_DEGREES'] = (0, 20)

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
GLOBALS['WANDB_PROJECT_NAME'] = "FG-NET dataset"

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
sys.path.append(GLOBALS['BASE_PATH'])
sys.path.append(os.path.join(GLOBALS['BASE_PATH'], "src"))
sys.path.append(os.path.join(GLOBALS['BASE_PATH'], "src/lib"))


# Importing the modules we are going to use
# ==============================================================================

import os
import logging
import gc
import functools
import math
import seaborn as sns
import time
import copy
import cProfile
import enum
from collections import Counter
from datetime import datetime
from pprint import pprint
from tqdm import tqdm
from typing import List



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
import wandb
import dotenv

# All concrete pieces we're using form sklearn
from sklearn.metrics import roc_auc_score, accuracy_score, silhouette_score

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
import lib.datasets as datasets

from lib.trainers import train_model_online
from lib.train_loggers import SilentLogger, TripletLoggerOffline, TripletLoggerOnline, TrainLogger, CompoundLogger, IntraClusterLogger, InterClusterLogger, RankAtKLogger, LocalRankAtKLogger
from lib.models import *
from lib.visualizations import *
from lib.models import ResNet18, LFWResNet18, LFWLightModel, NormalizedNet, RetrievalAdapter, FGLigthModel, CACDResnet18, CACDResnet50
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


# TODO -- DELETE
torch.autograd.set_detect_anomaly(True)

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

## Dataset downloading
# ==============================================================================
#
# - There is no package in pytorch / torchvision for this dataset
#  - So we download the dataset from an URL

datasets.download_fg_dataset(
    GLOBALS['DATA_PATH'],
    GLOBALS['DATASET_URL'],
    can_skip_download = True
)

datasets.download_cacd_dataset(
    GLOBALS['CACD_DATA_PATH'],
    GLOBALS['CACD_DATASET_URL'],
    can_skip_download = True,
    can_skip_extraction = True,
)

## Dataset loading
# ==============================================================================
#
# As mentioned before, we have to use our custom implementation for pytorch
# `Dataset` class


# Transformations that we want to apply when loading the data
# TODO -- we are using the same transform for both datasets
transform = transforms.Compose([

    # First, convert to a PIL image so we can resize
    transforms.ToPILImage(),

    # Some images are colored, other images are black and white
    # So convert all the images to black and white, but having three channels
    transforms.Grayscale(num_output_channels = 3),

    # Images have different shapes, so this normalization is needed
    transforms.Resize(GLOBALS['IMAGE_SHAPE'], antialias=True),

    # Pytorch only work with tensors
    transforms.ToTensor(),

    # Some normalization
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    ),
])

print("=> Wrapping the raw data into a `FGDataset")
fgnet_dataset = datasets.FGDataset(path = GLOBALS['IMAGE_DIR_PATH'], transform = transform)

print("=> Wrapping the raw data into `CACDDataset`")
cacd_dataset = datasets.CACDDataset(path = GLOBALS['CACD_IMAGE_DIR_PATH'], transform = transform)

## Splitting the dataset
# ==============================================================================

# This function returns a `WrappedSubset` so we can still have access to
# `targets` attribute. With `Subset` pytorch class we cannot do that
# This split function returns subsets with disjoint classes. That is to say,
# if there is one person in one dataset, that person cannot appear in the
# other datset. Thus, percentages may vary a little
print("=> Splitting the dataset")
train_dataset, validation_dataset = split_dataset.split_dataset_disjoint_classes(cacd_dataset, 0.8)
test_dataset = fgnet_dataset

print("--> Dataset sizes:")
print(f"\tTrain dataset: {len(train_dataset) / len(cacd_dataset) * 100}%")
print(f"\tValidation dataset: {len(validation_dataset) / len(cacd_dataset) * 100}%")
print("")

print("--> Logging sizes:")
print(f"\tTrain dataset: {len(train_dataset) * GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE']}")
print(f"\tValidation dataset: {len(validation_dataset) * GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE']}")
print("")

## Use our custom sampler
# ==============================================================================
#
# - We want to use custom `Sampler` to do $P-K$ sampling
#     - For each minibatch, select $P$ random classes. For each selected class, select $K$ random images
# - Thus, each minibatch has a size of $P \times K$

print("=> Putting the dataset into dataloaders")

# New dataloaders that use our custom sampler
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
    sampler = CustomSampler(
        GLOBALS['P'],
        GLOBALS['K'],
        train_dataset,
        avoid_failing = GLOBALS['AVOID_CUSTOM_SAMPLER_FAIL'],
    )
)

# TODO -- here I don't know if use default sampler or custom sampler
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
    sampler = CustomSampler(
        GLOBALS['P'],
        GLOBALS['K'],
        dataset = validation_dataset,
        avoid_failing = GLOBALS['AVOID_CUSTOM_SAMPLER_FAIL'],
    ),
)

# TODO -- here I don't know if use default sampler or custom sampler
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
    shuffle = True,
)


# Data augmentation
# ==============================================================================
#
#  - Sometimes we have a `K` value way too big. In that case, some classes might have
#  less than `K` images, and thus, they cannot be used.
#
#  - For tackling this problem, we can perform data augmentation to assure that every
#  class has at least `K` images
#
# - In this dataset is not a big problem, in contrast with what happens in the
#   LFW dataset
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

    print("=> Performing data augmentation")

    # Select the data augmentation mechanism
    AugmentationClass = LazyAugmentatedDataset if GLOBALS['LAZY_DATA_AUGMENTATION'] is True else AugmentatedDataset

    train_dataset_augmented = AugmentationClass(
        base_dataset = train_dataset,
        min_number_of_images = GLOBALS['K'],

        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        transform = transforms.Compose([

            # NOTE: We have normalized our images to be (3, 300, 300), so new
            # randomly generated images have to have the same shape
            transforms.RandomResizedCrop(size=GLOBALS['IMAGE_SHAPE'], antialias = True),
            transforms.RandomRotation(degrees=GLOBALS['ROTATE_AUGM_DEGREES']),
            transforms.RandomAutocontrast(),
        ])
    )

    validation_dataset_augmented = AugmentationClass(
        base_dataset = validation_dataset,
        min_number_of_images = GLOBALS['K'],

        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        transform = transforms.Compose([

            # NOTE: We have normalized our images to be (3, 300, 300), so new
            # randomly generated images have to have the same shape
            transforms.RandomResizedCrop(size=GLOBALS['IMAGE_SHAPE'], antialias = True),
            transforms.RandomRotation(degrees=GLOBALS['ROTATE_AUGM_DEGREES']),
            transforms.RandomAutocontrast(),
        ])
    )

    # TODO -- augmented dataset also for validation, in case we think that the
    #         CustomSampler is a good idea for the validation dataset
    # Save the augmented dataset to cache
    torch.save(train_dataset_augmented, GLOBALS['AUGMENTED_DATASET_CACHE_FILE'])

# Now put a loader in front of the augmented dataset
train_loader_augmented = torch.utils.data.DataLoader(
    train_dataset_augmented,
    batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
    sampler = CustomSampler(
        GLOBALS['P'],
        GLOBALS['K'],
        train_dataset_augmented,
        avoid_failing = GLOBALS['AVOID_CUSTOM_SAMPLER_FAIL'],
    )
)

validation_loader_augmented = torch.utils.data.DataLoader(
    validation_dataset_augmented,
    batch_size = GLOBALS['ONLINE_BATCH_SIZE'],
    sampler = CustomSampler(
        GLOBALS['P'],
        GLOBALS['K'],
        validation_dataset_augmented,
        avoid_failing = GLOBALS['AVOID_CUSTOM_SAMPLER_FAIL'],
    )
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
    del validation_dataset
    del validation_loader

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
# fold, and produces the loss value or other metric that we want to optimize
def loss_function(net: torch.nn.Module, validation_fold: DataLoader) -> float:
    return metrics.rank_accuracy(
        k = 1,
        data_loader = validation_fold,
        network = net,
        max_examples = len(validation_fold),
        fast_implementation = False,
    )

class TuningStrat(enum.Enum):
    HOLDOUT = "Hold Out"
    KFOLD = "K-Fold Cross Validation"

def objective(trial, implementation: TuningStrat):
    """
    Optuna function that is going to be used in the optimization process

    Depending on `implementation`, it computes the metric score to optimize in the following way:

    - TuningStrat.HOLDOUT: uses holdout method (training - validation). That is
                           to say, trains on training dataset, computes the
                           metric on the validation set
    - TuningStrat.KFOLD: uses k-fold cross validation for computing the metric
    """

    # Parameters that we are exploring
    # This is shared among underlying implementations
    p = trial.suggest_int("P", 2, 10)
    k = trial.suggest_int("K", 2, 10)
    net_election = trial.suggest_categorical(
        "Network",
        ["CACDResNet18", "CACDResNet50", "FGLightModel"]
    )
    normalization_election = trial.suggest_categorical(
        "UseNormalization", [True, False]
    )
    embedding_dimension = trial.suggest_int("Embedding Dimension", 1, 10)
    learning_rate = trial.suggest_float("Learning rate", 0, 0.001)
    softplus = trial.suggest_categorical("Use Softplus", [True, False])
    use_norm_penalty = trial.suggest_categorical("Use norm penalty", [True, False])

    use_gradient_clipping = trial.suggest_categorical(
        "UseGradientClipping", [True, False]
    )

    norm_penalty = None
    if use_norm_penalty is True:
        norm_penalty = trial.suggest_float("Norm penalty factor", 0.0001, 2.0)

    margin = None
    if softplus is False:
        margin = trial.suggest_float("Margin", 0.001, 1.0)

    gradient_clipping = None
    if use_gradient_clipping is True:
        gradient_clipping = trial.suggest_float("Gradient Clipping Value", 0.00001, 10.0)

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
        # Again, we want to end with the normalized shape
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=GLOBALS['IMAGE_SHAPE'], antialias = True),
            transforms.RandomRotation(degrees=GLOBALS['ROTATE_AUGM_DEGREES']),
            transforms.RandomAutocontrast(),
        ])
    )

    # Put some dataloaders
    # TODO -- esto en la implementacion original se hace en el `loader_generator`
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = p * k,
        sampler = CustomSampler(
            p,
            k,
            train_dataset,
            avoid_failing = GLOBALS['AVOID_CUSTOM_SAMPLER_FAIL'],
        ),
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
        # As we are computing Rank@k we need to augment both fold types
        #
        # NOTE -- this can affect the optimization process, but otherwise most
        # of the tries will fail because of this problem
        if fold_type is hptuning.FoldType.TRAIN_FOLD or fold_type is hptuning.FoldType.VALIDATION_FOLD:
            fold_dataset_augmented = LazyAugmentatedDataset(
                base_dataset = fold_dataset,
                min_number_of_images = k,

                # Remember that the trasformation has to be random type
                # Otherwise, we could end with a lot of repeated images
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=GLOBALS['IMAGE_SHAPE'], antialias = True),
                    transforms.RandomRotation(degrees=GLOBALS['ROTATE_AUGM_DEGREES']),
                    transforms.RandomAutocontrast(),
                ])

            )
        else:
            # We are always doing data augmentation
            # NOTE: this `if else` is useless, but for other target metrics might
            # be needed
            raise ValueError(f"Got {fold_type=} which is not a valid fold type")

        # In the same line, we only use our custom sampler for the training fold
        # Otherwise, we use a normal sampler
        # NOTE: depending on the `loss_function`, maybe validation loader has to
        # have our CustomSampler
        if fold_type is hptuning.FoldType.TRAIN_FOLD:
            loader = torch.utils.data.DataLoader(
                fold_dataset_augmented,
                batch_size = p * k,
                sampler = CustomSampler(
                    p,
                    k,
                    fold_dataset,
                    avoid_failing = GLOBALS['AVOID_CUSTOM_SAMPLER_FAIL'],
                )
            )
        elif fold_type is hptuning.FoldType.VALIDATION_FOLD:
            loader = torch.utils.data.DataLoader(
                fold_dataset_augmented,
                batch_size = p * k,
            )
        else:
            raise ValueError(f"{fold_type=} enum value is not managed in if elif construct!")

        # To avoid accessing loader.__len__ without computing the necessary data
        _ = loader.__iter__()

        return loader


    # Wrap the network in a lambda function so we can use it in `custom_cross_validation`
    def network_creator():

        # Model that we have chosen
        if net_election == "FGLightModel":
            net = FGLigthModel(embedding_dimension)
        elif net_election == "CACDResNet18":
            net = CACDResnet18(embedding_dimension)
        elif net_election == "CACDResNet50":
            net = CACDResnet50(embedding_dimension)
        else:
            err_msg = "Parameter `net_election` has not a valid value \n"
            err_msg += f"{net_election=}"
            raise Exception(err_msg)

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
        parameters["criterion"] = BatchHardTripletLoss(
            margin,
            use_softplus = softplus,
            use_gt_than_zero_mean = True
        )

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
            gradient_clipping = gradient_clipping,
            fail_fast = True,
        )

        return net

    # Train and evaluate the model to obtain a loss metric
    # This is where the two implementations differ considerably
    if implementation is TuningStrat.HOLDOUT:
        # Train the model, might fail
        net = network_creator()
        try:
            net = network_trainer(train_loader, net)
        except Exception as e:
            print(f"Error training the network, reason was: {e}")

            # Let optuna know that this set of parameters produced a failure in the
            # training process
            return None

        # Evaluate the model
        loss = loss_function(net, validation_loader_augmented)

    elif implementation is TuningStrat.KFOLD:

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
            print(f"Array of losses: {losses=}")
            print(f"Obtained loss (cross validation mean) is {losses.mean()=}")

        except Exception as e:

            # Show that cross validation failed for this combination
            msg = "Could not run succesfully k-fold cross validation for this combination of parameters\n"
            msg = msg + f"Error was: {e}\n"
            print(msg)
            file_logger.warn(msg)

            # Return None so optuna knows this trial failed
            return None

        # If everything went alright, return the mean of the loss
        loss = losses.mean()

    else:
        # This should never happen, but add a check in case we modify the enum
        # so we don't forget to modify this *"match"* statement
        raise Exception(f"Got invalid implementation enum\n{implementation=}")

    return loss


# Launch the hp tuning process
if GLOBALS['SKIP_HYPERPARAMTER_TUNING'] is False:

    # We want to chose the `objective` implementation to use. But optuna only
    # accepts functions with the shape `objective(trial)` so get a partial
    # function with the parameter `implementation chosen`
    strat = TuningStrat.HOLDOUT if GLOBALS['FAST_HP_TUNING'] is True else TuningStrat.KFOLD
    partial_objective = lambda trial: objective(trial, implementation = strat)


    print(f"🔎 Started hyperparameter tuning with {strat=}")
    print("")

    study = optuna.create_study(
        direction = "maximize",
        study_name = "Rank@1 optimization",
        storage = GLOBALS['OPTUNA_DATABASE'],
        load_if_exists = True
    )
    study.optimize(partial_objective, n_trials = GLOBALS['HYPERPARAMETER_TUNING_TRIES'])

    print("🔎 Hyperparameter tuning ended")
    print("")
    print(f"🔎 Best trial: {study.best_trial}")
    print("")


# Training of the model
# ==============================================================================

## Selecting the network and tweaking some parameters
# ==============================================================================


print("=> Selecting the network model")
net = None
if GLOBALS['NET_MODEL'] == "ResNet18":
    net = ResNet18(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "LightModel":
    net = LightModel(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "LFWResNet18":
    net = LFWResNet18(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "LFWLightModel":
    net = LFWLightModel(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "FGLightModel":
    net = FGLigthModel(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "CACDResNet18":
    net = CACDResnet18(GLOBALS['EMBEDDING_DIMENSION'])
elif GLOBALS['NET_MODEL'] == "CACDResNet50":
    net = CACDResnet50(GLOBALS['EMBEDDING_DIMENSION'])
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


print("=> Creating the training loggers that we are going to use")

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

rank_at_one_logger = RankAtKLogger(
    net = net,
    iterations = GLOBALS['LOGGING_ITERATIONS'],
    train_percentage = GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'],
    validation_percentage = GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'],
    k = 1
)

rank_at_k_logger = RankAtKLogger(
    net = net,
    iterations = GLOBALS['LOGGING_ITERATIONS'],
    train_percentage = GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'],
    validation_percentage = GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'],
    k = GLOBALS['ACCURACY_AT_K_VALUE']
)


local_rank_at_one_logger = LocalRankAtKLogger(
    net = net,
    iterations = GLOBALS['LOGGING_ITERATIONS'],
    train_percentage = GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'],
    validation_percentage = GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'],
    k = 1
)

local_rank_at_k_logger = LocalRankAtKLogger(
    net = net,
    iterations = GLOBALS['LOGGING_ITERATIONS'],
    train_percentage = GLOBALS['ONLINE_LOGGER_TRAIN_PERCENTAGE'],
    validation_percentage = GLOBALS['ONLINE_LOGGER_VALIDATION_PERCENTAGE'],
    k = GLOBALS['ACCURACY_AT_K_VALUE']
)

# Combine them in a single logger
logger = CompoundLogger([
    triplet_loss_logger,
    cluster_sizes_logger,
    intercluster_metrics_logger,
    rank_at_one_logger,
    rank_at_k_logger,
    local_rank_at_one_logger,
    local_rank_at_k_logger,
])


## Running the training loop
# ==============================================================================

# Check if we want to skip training
if GLOBALS['USE_CACHED_MODEL'] is False:

    # To measure the time it takes to train
    ts = time.time()

    # Run the training with the profiling
    if GLOBALS['PROFILE_TRAINING'] is True:
        _ = cProfile.run(
            f"""train_model_online(
                net = net,
                path = os.path.join(BASE_PATH, 'tmp'),
                parameters = parameters,
                train_loader = train_loader_augmented,
                validation_loader = validation_loader_augmented,
                name = NET_MODEL,
                logger = logger,
                snapshot_iterations = None,
                gradient_clipping = GRADIENT_CLIPPING,
            )""",
            GLOBALS['PROFILE_SAVE_FILE']
        )

    # Run the training without the profiling
    else:

        training_history = train_model_online(
            net = net,
            path = os.path.join(GLOBALS['BASE_PATH'], "tmp"),
            parameters = parameters,
            train_loader = train_loader_augmented,
            validation_loader = validation_loader_augmented,
            name = GLOBALS['NET_MODEL'],
            logger = logger,
            snapshot_iterations = None,
            gradient_clipping = GLOBALS['GRADIENT_CLIPPING'],
        )

    # Compute how long it took
    te = time.time()
    print(f"It took {te - ts} seconds to train")

    # Update the model cache
    filesystem.save_model(net, GLOBALS['MODEL_CACHE_FOLDER'], "online_model_cached")

# In case we skipped training, load the model from cache
else:

    # Choose the function to construct the new network
    if GLOBALS['NET_MODEL'] == "ResNet18":
        net_func = lambda: ResNet18(GLOBALS['EMBEDDING_DIMENSION'])
    elif GLOBALS['NET_MODEL'] == "LightModel":
        net_func = lambda: LightModel(GLOBALS['EMBEDDING_DIMENSION'])
    elif GLOBALS['NET_MODEL'] == "LFWResNet18":
        net_func = lambda: LFWResNet18(GLOBALS['EMBEDDING_DIMENSION'])
    elif GLOBALS['NET_MODEL'] == "LFWLightModel":
        net_func = lambda: LFWLightModel(GLOBALS['EMBEDDING_DIMENSION'])
    elif GLOBALS['NET_MODEL'] == "FGLightModel":
        net_func = lambda: FGLigthModel(GLOBALS['EMBEDDING_DIMENSION'])
    elif GLOBALS['NET_MODEL'] == "CADResNet18":
        net_func = lambda: CACDResnet18(GLOBALS['EMBEDDING_DIMENSION'])
    elif GLOBALS['NET_MODEL'] == "CACDResNet50":
        net_func = lambda: CACDResnet50(GLOBALS['EMBEDDING_DIMENSION'])
    else:
        raise Exception("Parameter 'NET_MODEL' has not a valid value")

    # Load the model from cache
    net = filesystem.load_model(
        os.path.join(GLOBALS['MODEL_CACHE_FOLDER'], "online_model_cached"),
        net_func
    )

    # Load the network in corresponding mem device (cpu -> ram, gpu -> gpu mem
    device = core.get_device()
    net.to(device)



# From this point, we won't perform training on the model
# So eval mode is set for better performance
net.eval()


# Model evaluation
# ==============================================================================


# Use the network to perform a retrieval task and compute rank@1 and rank@5 accuracy
with torch.no_grad():
    net.set_permute(False)

    train_rank_at_one = metrics.rank_accuracy(
        k = 1,
        data_loader = train_loader_augmented,
        network = net,
        max_examples = len(train_loader_augmented),
        fast_implementation = False
    )
    test_rank_at_one = metrics.rank_accuracy(
        k = 1,
        data_loader = test_loader,
        network = net,
        max_examples = len(test_loader),
        fast_implementation = False,
    )
    train_rank_at_five = metrics.rank_accuracy(
        k = 5,
        data_loader = train_loader_augmented,
        network = net,
        max_examples = len(train_loader_augmented),
        fast_implementation = False
    )
    test_rank_at_five = metrics.rank_accuracy(
        k = 5,
        data_loader = test_loader,
        network = net,
        max_examples = len(test_loader),
        fast_implementation = False
    )

    print(f"Train Rank@1 Accuracy: {train_rank_at_one}")
    print(f"Test Rank@1 Accuracy: {test_rank_at_one}")
    print(f"Train Rank@5 Accuracy: {train_rank_at_five}")
    print(f"Test Rank@5 Accuracy: {test_rank_at_five}")

    # Put this info in wandb
    wandb.log({
        "Final Train Rank@1 Accuracy": train_rank_at_one,
        "Final Test Rank@1 Accuracy": test_rank_at_one,
        "Final Train Rank@5 Accuracy": train_rank_at_five,
        "Final Test Rank@5 Accuracy": test_rank_at_five,
    })

    net.set_permute(True)


# Compute the the *silhouette* metric for the produced embedding, on
# train, validation and test set:
with torch.no_grad():

    net.set_permute(False)

    # Try to clean memory, because we can easily run out of memory
    # This provoke the notebook to crash, and all in-memory objects to be lost
    try_to_clean_memory()

    train_silh = metrics.silhouette(train_loader_augmented, net)
    print(f"Silhouette in training loader: {train_silh}")

    validation_silh = metrics.silhouette(validation_loader_augmented, net)
    print(f"Silhouette in validation loader: {validation_silh}")

    test_silh = metrics.silhouette(test_loader, net)
    print(f"Silhouette in test loader: {test_silh}")

    # Put this info in wandb
    wandb.log({
        "Final Training silh": train_silh,
        "Final Validation silh": validation_silh,
        "Final Test silh": test_silh
    })

    net.set_permute(True)


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

# See how it works on a small test set
with torch.no_grad():

    net.set_permute(False)

    # Show only `max_iterations` classifications
    counter = 0
    max_iterations = 20

    for img, img_class in test_dataset:
        predicted_class = classifier.predict(img)
        print(f"True label: {img_class}, predicted label: {predicted_class[0]}, correct: {img_class == predicted_class[0]}")

        counter += 1
        if counter == max_iterations: break

    net.set_permute(True)


# Plot of the embedding
# ==============================================================================
#
# - If the dimension of the embedding is 2, then we can plot how the transformation to a classificator works:
# - That logic is encoded in the `scatter_plot` method
with torch.no_grad():
    classifier.scatter_plot()
