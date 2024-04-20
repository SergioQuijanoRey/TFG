# MNIST
# ==============================================================================
#
# - Check `MNIST Notebook.ipynb` for:
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
GLOBALS["RUNNING_ENV"] = "ugr"

# Base path for the rest of paths defined in the notebook
GLOBALS["BASE_PATH"] = None
if GLOBALS["RUNNING_ENV"] == "local":
    GLOBALS["BASE_PATH"] = "./"
elif GLOBALS["RUNNING_ENV"] == "remote":
    GLOBALS["BASE_PATH"] = "/content/drive/MyDrive/Colab Notebooks/"
elif GLOBALS["RUNNING_ENV"] == "ugr":
    GLOBALS["BASE_PATH"] = "/mnt/homeGPU/squijano/TFG/"
else:
    raise Exception(f"RUNNING ENV is not valid, got value {GLOBALS['RUNNING_ENV']}")

# Path to our lib dir
GLOBALS["LIB_PATH"] = os.path.join(GLOBALS["BASE_PATH"], "lib")

# Path where we store training / test data
GLOBALS["DATA_PATH"] = os.path.join(GLOBALS["BASE_PATH"], "data")

# Dataset has images and metadata. Here we store the path to the img dir
GLOBALS["IMAGE_DIR_PATH"] = os.path.join(GLOBALS["DATA_PATH"], "FGNET/images")

# Dir with all cached models
# This cached models can be loaded from disk when training is skipped
GLOBALS["MODEL_CACHE_FOLDER"] = os.path.join(GLOBALS["BASE_PATH"], "cached_models")

# Cache for the augmented dataset
GLOBALS["AUGMENTED_DATASET_CACHE_FILE"] = os.path.join(
    GLOBALS["BASE_PATH"], "cached_augmented_dataset.pt"
)

# File where the logs are written
GLOBALS["LOGGING_FILE"] = os.path.join(GLOBALS["BASE_PATH"], "training.log")

# Binary file where the stats of the profiling are saved
GLOBALS["PROFILE_SAVE_FILE"] = os.path.join(
    GLOBALS["BASE_PATH"], "training_profile.stat"
)

GLOBALS["OPTUNA_DATABASE"] = f"sqlite:///{GLOBALS['BASE_PATH']}/hp_tuning_optuna.db"

## ML parameters
# ==============================================================================
#
# - Parameters related to machine learning
# - For example, batch sizes, learning rates, ...


# Parameters of P-K sampling
GLOBALS["P"] = 8  # Number of classes used in each minibatch
GLOBALS["K"] = 4  # Number of images sampled for each selected class

# Batch size for online training
# We can use `P * K` as batch size. Thus, minibatches will be
# as we expect in P-K sampling.
#
# But we can use `n * P * K`. Thus, minibatches will be n P-K sampling
# minibatche concatenated together
# Be careful when doing this because it can be really slow, and there is no
# clear reason to do this
GLOBALS["ONLINE_BATCH_SIZE"] = GLOBALS["P"] * GLOBALS["K"]

# Training epochs
GLOBALS["TRAINING_EPOCHS"] = 1

# Learning rate for hard triplets, online training
GLOBALS["ONLINE_LEARNING_RATE"] = 0.01

# How many single elements we want to see before logging
# It has to be a multiple of P * K, otherwise `should_log` would return always
# false as `it % LOGGING_ITERATIONS != 0` always
#
# `LOGGING_ITERATIONS = P * K * n` means we log after seeing `n` P-K sampled
# minibatches
#  GLOBALS['LOGGING_ITERATIONS'] = GLOBALS['P'] * GLOBALS['K'] * 500
GLOBALS["LOGGING_ITERATIONS"] = GLOBALS["P"] * GLOBALS["K"] * 1_000

# Which percentage of the training and validation set we want to use for the logging
GLOBALS["ONLINE_LOGGER_TRAIN_PERCENTAGE"] = 0.005
GLOBALS["ONLINE_LOGGER_VALIDATION_PERCENTAGE"] = 0.005

# Choose which model we're going to use
# Can be "ResNet18", "LightModel", "LFWResNet18", "LFWLightModel", "FGLightModel",
#        "CACDResNet18", "CACDResNet50"
GLOBALS["NET_MODEL"] = "LightModel"

# Epochs used in k-Fold Cross validation
# k-Fold Cross validation used for parameter exploration
# TODO -- delete this, we are going to perform a search in the number of epochs
GLOBALS["HYPERPARAMETER_TUNING_EPOCHS"] = 1

# Number of tries in the optimization process
# We are using optuna, so we try `HYPERPARAMETER_TUNING_TRIES` times with different
# hyperparameter configurations
GLOBALS["HYPERPARAMETER_TUNING_TRIES"] = 300

# Wether to use the validation set in the hp tuning process or to use k-fold
# cross validation (which is more robust but way slower)
GLOBALS["FAST_HP_TUNING"] = True

# Number of folds used in k-fold Cross Validation
GLOBALS["NUMBER_OF_FOLDS"] = 2

# Margin used in the loss function
GLOBALS["MARGIN"] = 0.840

# Dim of the embedding calculated by the network
GLOBALS["EMBEDDING_DIMENSION"] = 5

# Number of neighbours considered in K-NN
# K-NN used for transforming embedding task to classification task
GLOBALS["NUMBER_NEIGHBOURS"] = 4

# Batch Triplet Loss Function
# This way we can choose among "hard", "all"
GLOBALS["BATCH_TRIPLET_LOSS_FUNCTION"] = "hard"

# Whether or not use softplus loss function instead of vanilla triplet loss
GLOBALS["USE_SOFTPLUS_LOSS"] = False

# Count all sumamnds in the mean loss or only those summands greater than zero
GLOBALS["USE_GT_ZERO_MEAN_LOSS"] = True

# Wether or not use lazy computations in the data augmentation
GLOBALS["LAZY_DATA_AUGMENTATION"] = True

# Wether or not fail when calling `CustomSampler.__len__` without having previously
# computed the index list
GLOBALS["AVOID_CUSTOM_SAMPLER_FAIL"] = True

# Where or not add penalty term to the loss function
GLOBALS["ADD_NORM_PENALTY"] = False

# If we add that penalty term, which scaling factor to use
GLOBALS["PENALTY_FACTOR"] = 0.6

# If we want to wrap our model into a normalizer
# That wrapper divides each vector by its norm, thus, forcing norm 1 on each vector
GLOBALS["NORMALIZED_MODEL_OUTPUT"] = False

# If its None, we do not perform gradient clipping
# If its a Float value, we perform gradient clipping, using that value as a
# parameter for the max norm
GLOBALS["GRADIENT_CLIPPING"] = None

# Number of candidates that we are going to consider in the retrieval task,
# used in the Rank@K accuracy metric
# We use k = 1 and k = this value
GLOBALS["ACCURACY_AT_K_VALUE"] = 5

# Images in this dataset have different shapes. So this parameter fixes one shape
# so we can normalize the images to have the same shape
GLOBALS["IMAGE_SHAPE"] = (200, 200)

# Degrees that we are going to use in data augmentation rotations
GLOBALS["ROTATE_AUGM_DEGREES"] = (0, 20)

## Section parameters
# ==============================================================================

# - Flags to choose if some sections will run or not
# - This way we can skip some heavy computations when not needed

# Skip hyper parameter tuning for online training
GLOBALS["SKIP_HYPERPARAMTER_TUNING"] = True

# Skip training and use a cached model
# Useful for testing the embedding -> classifier transformation
# Thus, when False training is not computed and a cached model
# is loaded from disk
# Cached models are stored in `MODEL_CACHE_FOLDER`
GLOBALS["USE_CACHED_MODEL"] = False

# Skip data augmentation and use the cached augmented dataset
GLOBALS["USE_CACHED_AUGMENTED_DATASET"] = False

# Most of the time we're not exploring the data, but doing
# either hyperparameter settings or training of the model
# So if we skip this step we can start the process faster
GLOBALS["SKIP_EXPLORATORY_DATA_ANALYSYS"] = True

# Wether or not profile the training
# This should be False most of the times
# Note that profiling adds a significant overhead to the training
GLOBALS["PROFILE_TRAINING"] = False


## WANDB Parameters
# ==============================================================================

from datetime import datetime

# Name for the project
# One project groups different runs
GLOBALS["WANDB_PROJECT_NAME"] = "MNIST dataset"

# Name for this concrete run
# I don't care too much about it, because wandb tracks the parameters we use
# in this run (see "Configuration for Weights and Biases" section)
GLOBALS["WANDB_RUN_NAME"] = str(datetime.now())


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
# TODO -- ADAM's script had 1 and the time difference was not big
GLOBALS["NUM_WORKERS"] = 4

# Fix random seed to make reproducible results
GLOBALS["RANDOM_SEED"] = 123456789

# Add some paths to PYTHONPATH
# ==============================================================================

# Python paths are difficult to manage
# In this script we can do something like:
# `import lib.core as core` and that's fine
# But in lib code we cannot import properly the modules

import sys

sys.path.append(GLOBALS["BASE_PATH"])
sys.path.append(os.path.join(GLOBALS["BASE_PATH"], "src"))
sys.path.append(os.path.join(GLOBALS["BASE_PATH"], "src/lib"))


# Importing the modules we are going to use
# ==============================================================================

import copy
import cProfile
import enum
import functools
import gc
import logging
import math
import os
import time
from collections import Counter
from datetime import datetime
from pprint import pprint
from typing import List

import dotenv
import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
# For using pre-trained ResNets
import torchvision.models as models
import torchvision.transforms as transforms
# All concrete pieces we're using form sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, silhouette_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Now that files are loaded, we can import pieces of code
import lib.core as core
import lib.data_augmentation as data_augmentation
import lib.datasets as datasets
import lib.embedding_to_classifier as embedding_to_classifier
import lib.filesystem as filesystem
import lib.hyperparameter_tuning as hptuning
import lib.loss_functions as loss_functions
import lib.metrics as metrics
import lib.sampler as sampler
import lib.split_dataset as split_dataset
import lib.trainers as trainers
import lib.utils as utils
import wandb
from lib.data_augmentation import AugmentatedDataset, LazyAugmentatedDataset
from lib.embedding_to_classifier import EmbeddingToClassifier
from lib.loss_functions import (AddSmallEmbeddingPenalization,
                                BatchAllTripletLoss, BatchHardTripletLoss,
                                MeanTripletBatchTripletLoss)
from lib.models import *
from lib.models import (CACDResnet18, CACDResnet50, FGLigthModel,
                        LFWLightModel, LFWResNet18, NormalizedNet, ResNet18,
                        RetrievalAdapter)
from lib.sampler import CustomSampler
from lib.train_loggers import (CompoundLogger, InterClusterLogger,
                               IntraClusterLogger, LocalRankAtKLogger,
                               RankAtKLogger, SilentLogger, TrainLogger,
                               TripletLoggerOffline, TripletLoggerOnline)
from lib.trainers import train_model_online
from lib.visualizations import *

# Server security check
# ==============================================================================
#
# - Sometimes UGR's server does not provide GPU access
# - In that case, fail fast so we start ASAP debugging the problem

if GLOBALS["RUNNING_ENV"] == "ugr" and torch.cuda.is_available() is False:
    raise Exception(
        "`torch.cuda.is_available()` returned false, so we dont have access to GPU's"
    )


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
file_handler = logging.FileHandler(GLOBALS["LOGGING_FILE"])
file_handler.setFormatter(formatter)
file_logger.addHandler(file_handler)

# 'application' code
file_logger.debug("debug message")


# Configuration for Weigths and Biases
# ==============================================================================
#
# - We're going to use `wandb` for tracking the training of the models
# - We use our `GLOBALS` dict to init wandb, that is going to keep track of all
# of that parameters

# If we're running in UGR's servers, we need to set some ENV vars
# Otherwise, wandb is going to write to dirs that it has no access
# Also, pytorch tries to save pretrained models in the home folder
if GLOBALS["RUNNING_ENV"] == "ugr":
    print("-> Changing dir env values")
    utils.change_dir_env_vars(base_path=GLOBALS["BASE_PATH"])
    print("-> Changing done!")
    print("")

    print("-> Login again to WANDB")
    utils.login_wandb()
    print("-> Login done!")
    print("")

# Init the wandb tracker
# We need to do this before `wandb.login`
wandb.init(
    project=GLOBALS["WANDB_PROJECT_NAME"],
    name=GLOBALS["WANDB_RUN_NAME"],
    config=GLOBALS,
)

# Functions that we are going to use
# ==============================================================================


def show_learning_curve(training_history: dict):
    # Take two learning curves
    loss = training_history["loss"]
    val_loss = training_history["val_loss"]

    # Move the lists to cpu, because that's what matplotlib needs
    loss = [loss_el.cpu() for loss_el in loss]
    val_loss = [val_loss_el.cpu() for val_loss_el in val_loss]

    # Show graphics
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(["Training loss", "Validation loss"])
    plt.show()


def try_to_clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


# Import MNIST dataset
# ==============================================================================


mean, std = 0.1307, 0.3081

train_dataset = torchvision.datasets.MNIST(
    "../data/MNIST",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    ),
)
test_dataset = torchvision.datasets.MNIST(
    "../data/MNIST",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    ),
)


## Use our custom sampler
# ==============================================================================

from adambielski_lib import datasets as adamdatasets

print("=> Putting the dataset into dataloaders")


# TODO -- ADAM -- use our global values
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = adamdatasets.BalancedBatchSampler(
    train_dataset.train_labels, n_classes=10, n_samples=25
)
test_batch_sampler = adamdatasets.BalancedBatchSampler(
    test_dataset.test_labels, n_classes=10, n_samples=25
)

cuda = torch.cuda.is_available()
kwargs = {"num_workers": 4, "pin_memory": True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_sampler=train_batch_sampler, **kwargs
)
online_test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_sampler=test_batch_sampler, **kwargs
)


# Choose the loss function to use
# ==============================================================================

from adambielski_lib import losses as adamlosses
from adambielski_lib import utils as adamutils

margin = 1.0
batch_loss_function = adamlosses.OnlineTripletLoss(
    margin, adamutils.RandomNegativeTripletSelector(margin)
)


# Select the network that we are going to use
# ==============================================================================

from adambielski_lib import networks as adamnetworks

net = adamnetworks.EmbeddingNet()
if cuda:
    net = net.cuda()

# Training of the model
# ==============================================================================

from adambielski_lib import metrics as adammetrics
from adambielski_lib import trainer as adamtrainer

lr = 1e-3
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

# TODO -- ADAM -- What is this?
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

# TODO -- ADAM -- Use our globals
n_epochs = 20
log_interval = 50

adamtrainer.fit(
    online_train_loader,
    online_test_loader,
    net,
    batch_loss_function,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    metrics=[adammetrics.AverageNonzeroTripletsMetric()],
)

# Evaluate the model
# ==============================================================================

mnist_classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
    plt.figure(figsize=(10, 10))
    for i in range(10):
        inds = np.where(targets == i)[0]
        plt.scatter(
            embeddings[inds, 0], embeddings[inds, 1], alpha=0.5, color=colors[i]
        )
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(mnist_classes)


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k : k + len(images)] = (
                model.get_embedding(images).data.cpu().numpy()
            )
            labels[k : k + len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


train_embeddings_otl, train_labels_otl = extract_embeddings(online_train_loader, net)
plot_embeddings(train_embeddings_otl, train_labels_otl)
val_embeddings_otl, val_labels_otl = extract_embeddings(online_test_loader, net)
plot_embeddings(val_embeddings_otl, val_labels_otl)

# TODO -- ADAM -- use our loggers in the training
#  ## Defining the loggers we want to use
#  # ==============================================================================


#  print("=> Creating the training loggers that we are going to use")

#  # Define the loggers we want to use
#  triplet_loss_logger = TripletLoggerOnline(
#      net=net,
#      iterations=GLOBALS["LOGGING_ITERATIONS"],
#      loss_func=parameters["criterion"],
#      train_percentage=GLOBALS["ONLINE_LOGGER_TRAIN_PERCENTAGE"],
#      validation_percentage=GLOBALS["ONLINE_LOGGER_VALIDATION_PERCENTAGE"],
#      greater_than_zero=GLOBALS["USE_GT_ZERO_MEAN_LOSS"],
#  )

#  cluster_sizes_logger = IntraClusterLogger(
#      net=net,
#      iterations=GLOBALS["LOGGING_ITERATIONS"],
#      train_percentage=GLOBALS["ONLINE_LOGGER_TRAIN_PERCENTAGE"],
#      validation_percentage=GLOBALS["ONLINE_LOGGER_VALIDATION_PERCENTAGE"],
#  )

#  intercluster_metrics_logger = InterClusterLogger(
#      net=net,
#      iterations=GLOBALS["LOGGING_ITERATIONS"],
#      train_percentage=GLOBALS["ONLINE_LOGGER_TRAIN_PERCENTAGE"],
#      validation_percentage=GLOBALS["ONLINE_LOGGER_VALIDATION_PERCENTAGE"],
#  )

#  rank_at_one_logger = RankAtKLogger(
#      net=net,
#      iterations=GLOBALS["LOGGING_ITERATIONS"],
#      train_percentage=GLOBALS["ONLINE_LOGGER_TRAIN_PERCENTAGE"],
#      validation_percentage=GLOBALS["ONLINE_LOGGER_VALIDATION_PERCENTAGE"],
#      k=1,
#  )

#  rank_at_k_logger = RankAtKLogger(
#      net=net,
#      iterations=GLOBALS["LOGGING_ITERATIONS"],
#      train_percentage=GLOBALS["ONLINE_LOGGER_TRAIN_PERCENTAGE"],
#      validation_percentage=GLOBALS["ONLINE_LOGGER_VALIDATION_PERCENTAGE"],
#      k=GLOBALS["ACCURACY_AT_K_VALUE"],
#  )


#  local_rank_at_one_logger = LocalRankAtKLogger(
#      net=net,
#      iterations=GLOBALS["LOGGING_ITERATIONS"],
#      train_percentage=GLOBALS["ONLINE_LOGGER_TRAIN_PERCENTAGE"],
#      validation_percentage=GLOBALS["ONLINE_LOGGER_VALIDATION_PERCENTAGE"],
#      k=1,
#  )

#  local_rank_at_k_logger = LocalRankAtKLogger(
#      net=net,
#      iterations=GLOBALS["LOGGING_ITERATIONS"],
#      train_percentage=GLOBALS["ONLINE_LOGGER_TRAIN_PERCENTAGE"],
#      validation_percentage=GLOBALS["ONLINE_LOGGER_VALIDATION_PERCENTAGE"],
#      k=GLOBALS["ACCURACY_AT_K_VALUE"],
#  )

#  # Combine them in a single logger
#  logger = CompoundLogger(
#      [
#          triplet_loss_logger,
#          cluster_sizes_logger,
#          intercluster_metrics_logger,
#          rank_at_one_logger,
#          rank_at_k_logger,
#          local_rank_at_one_logger,
#          local_rank_at_k_logger,
#      ]
#  )


# TODO -- run our training loop
#  ## Running the training loop
#  # ==============================================================================

#  # Check if we want to skip training
#  if GLOBALS["USE_CACHED_MODEL"] is False:
#      # To measure the time it takes to train
#      ts = time.time()

#      # Run the training with the profiling
#      if GLOBALS["PROFILE_TRAINING"] is True:
#          _ = cProfile.run(
#              f"""train_model_online(
#                  net = net,
#                  path = os.path.join(BASE_PATH, 'tmp'),
#                  parameters = parameters,
#                  train_loader = train_loader_augmented,
#                  validation_loader = validation_loader_augmented,
#                  name = NET_MODEL,
#                  logger = logger,
#                  snapshot_iterations = None,
#                  gradient_clipping = GRADIENT_CLIPPING,
#              )""",
#              GLOBALS["PROFILE_SAVE_FILE"],
#          )

#      # Run the training without the profiling
#      else:
#          training_history = train_model_online(
#              net=net,
#              path=os.path.join(GLOBALS["BASE_PATH"], "tmp"),
#              parameters=parameters,
#              train_loader=train_loader_augmented,
#              validation_loader=validation_loader_augmented,
#              name=GLOBALS["NET_MODEL"],
#              logger=logger,
#              snapshot_iterations=None,
#              gradient_clipping=GLOBALS["GRADIENT_CLIPPING"],
#          )

#      # Compute how long it took
#      te = time.time()
#      print(f"It took {te - ts} seconds to train")

#      # Update the model cache
#      filesystem.save_model(net, GLOBALS["MODEL_CACHE_FOLDER"], "online_model_cached")

#  # In case we skipped training, load the model from cache
#  else:
#      # Choose the function to construct the new network
#      if GLOBALS["NET_MODEL"] == "ResNet18":
#          net_func = lambda: ResNet18(GLOBALS["EMBEDDING_DIMENSION"])
#      elif GLOBALS["NET_MODEL"] == "LightModel":
#          net_func = lambda: LightModel(GLOBALS["EMBEDDING_DIMENSION"])
#      elif GLOBALS["NET_MODEL"] == "LFWResNet18":
#          net_func = lambda: LFWResNet18(GLOBALS["EMBEDDING_DIMENSION"])
#      elif GLOBALS["NET_MODEL"] == "LFWLightModel":
#          net_func = lambda: LFWLightModel(GLOBALS["EMBEDDING_DIMENSION"])
#      elif GLOBALS["NET_MODEL"] == "FGLightModel":
#          net_func = lambda: FGLigthModel(GLOBALS["EMBEDDING_DIMENSION"])
#      elif GLOBALS["NET_MODEL"] == "CADResNet18":
#          net_func = lambda: CACDResnet18(GLOBALS["EMBEDDING_DIMENSION"])
#      elif GLOBALS["NET_MODEL"] == "CACDResNet50":
#          net_func = lambda: CACDResnet50(GLOBALS["EMBEDDING_DIMENSION"])
#      else:
#          raise Exception("Parameter 'NET_MODEL' has not a valid value")

#      # Load the model from cache
#      net = filesystem.load_model(
#          os.path.join(GLOBALS["MODEL_CACHE_FOLDER"], "online_model_cached"), net_func
#      )

#      # Load the network in corresponding mem device (cpu -> ram, gpu -> gpu mem
#      device = core.get_device()
#      net.to(device)


#  # From this point, we won't perform training on the model
#  # So eval mode is set for better performance
#  net.eval()


# TODO -- ADAM -- Run our model evaluation
#  # Model evaluation
#  # ==============================================================================


#  # Use the network to perform a retrieval task and compute rank@1 and rank@5 accuracy
#  with torch.no_grad():
#      net.set_permute(False)

#      train_rank_at_one = metrics.rank_accuracy(
#          k=1,
#          data_loader=train_loader_augmented,
#          network=net,
#          max_examples=len(train_loader_augmented),
#          fast_implementation=False,
#      )
#      test_rank_at_one = metrics.rank_accuracy(
#          k=1,
#          data_loader=test_loader,
#          network=net,
#          max_examples=len(test_loader),
#          fast_implementation=False,
#      )
#      train_rank_at_five = metrics.rank_accuracy(
#          k=5,
#          data_loader=train_loader_augmented,
#          network=net,
#          max_examples=len(train_loader_augmented),
#          fast_implementation=False,
#      )
#      test_rank_at_five = metrics.rank_accuracy(
#          k=5,
#          data_loader=test_loader,
#          network=net,
#          max_examples=len(test_loader),
#          fast_implementation=False,
#      )

#      print(f"Train Rank@1 Accuracy: {train_rank_at_one}")
#      print(f"Test Rank@1 Accuracy: {test_rank_at_one}")
#      print(f"Train Rank@5 Accuracy: {train_rank_at_five}")
#      print(f"Test Rank@5 Accuracy: {test_rank_at_five}")

#      # Put this info in wandb
#      wandb.log(
#          {
#              "Final Train Rank@1 Accuracy": train_rank_at_one,
#              "Final Test Rank@1 Accuracy": test_rank_at_one,
#              "Final Train Rank@5 Accuracy": train_rank_at_five,
#              "Final Test Rank@5 Accuracy": test_rank_at_five,
#          }
#      )

#      net.set_permute(True)


#  # Compute the the *silhouette* metric for the produced embedding, on
#  # train, validation and test set:
#  with torch.no_grad():
#      net.set_permute(False)

#      # Try to clean memory, because we can easily run out of memory
#      # This provoke the notebook to crash, and all in-memory objects to be lost
#      try_to_clean_memory()

#      train_silh = metrics.silhouette(train_loader_augmented, net)
#      print(f"Silhouette in training loader: {train_silh}")

#      validation_silh = metrics.silhouette(validation_loader_augmented, net)
#      print(f"Silhouette in validation loader: {validation_silh}")

#      test_silh = metrics.silhouette(test_loader, net)
#      print(f"Silhouette in test loader: {test_silh}")

#      # Put this info in wandb
#      wandb.log(
#          {
#              "Final Training silh": train_silh,
#              "Final Validation silh": validation_silh,
#              "Final Test silh": test_silh,
#          }
#      )

#      net.set_permute(True)


#  # Show the "criterion" metric on test set
#  with torch.no_grad():
#      net.set_permute(False)

#      core.test_model_online(net, test_loader, parameters["criterion"], online=True)

#      net.set_permute(True)


#  # Now take the classifier from the embedding and use it to compute some classification metrics:
#  with torch.no_grad():
#      # Try to clean memory, because we can easily run out of memory
#      # This provoke the notebook to crash, and all in-memory objects to be lost
#      try_to_clean_memory()

#      # With hopefully enough memory, try to convert the embedding to a classificator
#      classifier = EmbeddingToClassifier(
#          net,
#          k=GLOBALS["NUMBER_NEIGHBOURS"],
#          data_loader=train_loader_augmented,
#          embedding_dimension=GLOBALS["EMBEDDING_DIMENSION"],
#      )

#  # See how it works on a small test set
#  with torch.no_grad():
#      net.set_permute(False)

#      # Show only `max_iterations` classifications
#      counter = 0
#      max_iterations = 20

#      for img, img_class in test_dataset:
#          predicted_class = classifier.predict(img)
#          print(
#              f"True label: {img_class}, predicted label: {predicted_class[0]}, correct: {img_class == predicted_class[0]}"
#          )

#          counter += 1
#          if counter == max_iterations:
#              break

#      net.set_permute(True)


# TODO -- ADAM -- run our plot of the embedding
#  # Plot of the embedding
#  # ==============================================================================
#  #
#  # - If the dimension of the embedding is 2, then we can plot how the transformation to a classificator works:
#  # - That logic is encoded in the `scatter_plot` method
#  with torch.no_grad():
#      classifier.scatter_plot()
