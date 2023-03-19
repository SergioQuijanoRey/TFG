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
P = 50    # Number of classes used in each minibatch
K = 3     # Number of images sampled for each selected class

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
TRAINING_EPOCHS = 10

# Learning rate for hard triplets, online training
ONLINE_LEARNING_RATE = 0.001

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
NET_MODEL = "LFWLightModel"

# Epochs used in k-Fold Cross validation
# k-Fold Cross validation used for parameter exploration
HYPERPARAMETER_TUNING_EPOCHS = 7

# Number of folds used in k-fold Cross Validation
NUMBER_OF_FOLDS = 4

# Margin used in the loss function
MARGIN = 0.5

# Dim of the embedding calculated by the network
EMBEDDING_DIMENSION = 3

# Number of neighbours considered in K-NN
# K-NN used for transforming embedding task to classification task
NUMBER_NEIGHBOURS = 4

# Batch Triplet Loss Function
# This way we can choose among "hard", "all"
BATCH_TRIPLET_LOSS_FUNCTION = "hard"

# Whether or not use softplus loss function instead of vanilla triplet loss
USE_SOFTPLUS_LOSS = False

# Count all sumamnds in the mean loss or only those summands greater than zero
USE_GT_ZERO_MEAN_LOSS = True

# Wether or not use lazy computations in the data augmentation
LAZY_DATA_AUGMENTATION = True

# Where or not add penalty term to the loss function
ADD_NORM_PENALTY = True

# If we add that penalty term, which scaling factor to use
PENALTY_FACTOR = 1.0


## Section parameters
# ==============================================================================

# - Flags to choose if some sections will run or not
# - This way we can skip some heavy computations when not needed


# Skip hyper parameter tuning for online training
SKIP_HYPERPARAMTER_TUNING = True

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
import src.lib.split_dataset as split_dataset

from src.lib.trainers import train_model_offline, train_model_online
from src.lib.train_loggers import SilentLogger, TripletLoggerOffline, TripletLoggerOnline, TrainLogger, CompoundLogger, IntraClusterLogger, InterClusterLogger
from src.lib.models import *
from src.lib.visualizations import *
from src.lib.models import ResNet18, LFWResNet18, LFWLightModel
from src.lib.loss_functions import MeanTripletBatchTripletLoss, BatchHardTripletLoss, BatchAllTripletLoss, AddSmallEmbeddingPenalization
from src.lib.embedding_to_classifier import EmbeddingToClassifier
from src.lib.sampler import CustomSampler
from src.lib.data_augmentation import AugmentatedDataset, LazyAugmentatedDataset


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

# If we're running in UGR's servers, we need to set some ENV vars
# Otherwise, wandb is going to write to dirs that it has no access
if RUNNING_ENV == "ugr":
    # TODO -- use `utils.wandb_log_and_set_env_vars()`

    base_path = BASE_PATH

    print("-> Changing WANDB env values")

    os.environ["WANDB_CONFIG_DIR"] = os.path.join(base_path, "wandb_config_dir_testing")
    os.environ["WANDB_CACHE_DIR"] = os.path.join(base_path, "wandb_cache_dir_testing")
    os.environ["WANDB_DIR"] = os.path.join(base_path, "wandb_dir_testing")
    os.environ["WANDB_DATA_DIR"] = os.path.join(base_path, "wandb_datadir_testing")

    print("-> Changing done!")
    print("")

    print("-> Loging to wandb using key stored in `.env`")
    dotenv.load_dotenv()
    wandb.login(key = os.environ["WANDB_API_KEY"])
    print("-> Loging done")
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


# Exploratory Data Analysis
# ==============================================================================

## Show some images from the dataset
# ==============================================================================
#
# Show some images with their classes, to verify that data loading was properly done:


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:

    imgs_to_show = 5

    for i, (img, label) in enumerate(train_dataset):

        # Show information about the image before plotting it
        print(f"Img label is: {label}")

        # Plot the img
        # TODO -- figure out a function to do propper img showing
        #img = img.reshape((250, 250, 3))
        #show_img(img, color_format_range = (-1.0, 1.0))
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

        # Stop the loop
        if i == imgs_to_show:
            break


# ## Show the sizes of the datasets


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    print(f"Train dataset: {len(train_dataset)}")
    print(f"Validation dataset: {len(validation_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")


# ## Show some of the triplets that we generated with our custom sampler:


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    # We show the triplets associated to only two classes
    triplets_to_show = 2 * K

    # TODO -- plot these images grouped in rows

    counter = 0
    finished = False
    for (imgs, labels) in train_loader:
        if finished is True:
            break

        for img, label in zip(imgs, labels):
            print(f"Current label is {label}")
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
            counter = counter + 1

            if counter >= triplets_to_show:
                finished = True
                break



# ## Explore how many images each class has
#
# Not all the classes have the same amount of images associated. So we plot the distribution of how many images has each class:


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    plot_how_many_images_per_class(
        train_dataset,
        cut = 25,
        fig_size = (10, 10)
    )


# We can see that most of the classes have only one image associated. Almost all classes have less than 10 images associated. So we have to develop a mechanism to do `P-K` sampling with a big enough value of `K`.

# ## Let's see how many classes have at least `K` images
#
# - This is important, because it shows us how many classes can be used in training using *P-K* sampling without any modification to the pipeline


def how_many_classes_have_at_least_K_images(dataset: torch.utils.data.Dataset, K: int):
    # Get the dict with class -> number of images of that class
    how_many_images_per_class = Counter(dataset.targets)

    # Get a list of classes that have at least P images
    # Use list comprehension with filtering over prev Counter dict
    classes_with_at_least_K_images = [
        curr_class
        for curr_class, curr_value in how_many_images_per_class.items()
        if curr_value >= K
    ]

    # Show some stats
    n = len(classes_with_at_least_K_images)
    print(f"There are {n} classes with at least {K} images")
    print(f"That represents the {n / len(how_many_images_per_class) * 100:.2f}% of the initial classes")


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    how_many_classes_have_at_least_K_images(train_dataset, K)


# # Data augmentation
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

# ## Augmentation of the dataset


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


# ## Remove previous datasets
#
# - If we're not doing hyperparameter tuning, we don't need to hold previous dataset and dataloader


if SKIP_HYPERPARAMTER_TUNING is True:
    # We are not using the old dataset and dataloader
    # So delete them to try to have as much RAM as we can
    # Otherwise, train will crash due to lack of RAM
    del train_dataset
    del train_loader

    try_to_clean_memory()


# ## Repeat some basic EDA on augmented dataset

# ### Show some images from the dataset
#
# Show some images with their classes, to verify that data loading was properly done:


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    imgs_to_show = 5

    for i, (img, label) in enumerate(train_dataset_augmented):

        # Show information about the image before plotting it
        print(f"Img label is: {label}")

        # Plot the img
        # TODO -- figure out a function to do propper img showing
        #img = img.reshape((250, 250, 3))
        #show_img(img, color_format_range = (-1.0, 1.0))
        plt.imshow(img.permute(1, 2, 0))
        plt.show()

        # Stop the loop
        if i == imgs_to_show:
            break


# ### Show the sizes of the datasets


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    print(f"Train dataset: {len(train_dataset_augmented)}")
    print(f"Validation dataset: {len(validation_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")


# ### Show some of the triplets that we generated with our custom sampler:


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    # We show the triplets associated to only two classes
    triplets_to_show = 2 * K

    # TODO -- plot these images grouped in rows

    counter = 0
    finished = False
    for (imgs, labels) in train_loader_augmented:
        if finished is True:
            break

        for img, label in zip(imgs, labels):
            print(f"Current label is {label}")
            plt.imshow(img.permute(1, 2, 0))
            plt.show()
            counter = counter + 1

            if counter >= triplets_to_show:
                finished = True
                break



# ### Explore how many images each class has
#
# Not all the classes have the same amount of images associated. So we plot the distribution of how many images has each class:


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    plot_how_many_images_per_class(
        train_dataset_augmented,
        cut = 25,
        fig_size = (10, 10)
    )


# We can see that most of the classes have only one image associated. Almost all classes have less than 10 images associated. So we have to develop a mechanism to do `P-K` sampling with a big enough value of `K`.

# ### Let's see how many classes have at least `P` images
#
# - This is important, because it shows us how many classes can be used in training using *P-K* sampling without any modification to the pipeline


if SKIP_EXPLORATORY_DATA_ANALYSYS is False:
    how_many_classes_have_at_least_K_images(train_dataset_augmented, K)


# # Choose the loss function to use
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


# # Choose wheter to add embedding norm or not


if ADD_NORM_PENALTY:
    batch_loss_function = AddSmallEmbeddingPenalization(
        base_loss = batch_loss_function,
        penalty_factor = PENALTY_FACTOR,
    )


# # Hyperparameter tuning


# TODO -- translate to english
# TODO -- move to lib
def custom_cross_validation(
        net: torch.nn.Module,
        parameters,
        train_dataset,
        k
    ):
    """
    Perform k-fold cross validation
    """

    # Definimos la forma en la que vamos a hacer el split de los folds
    ss = ShuffleSplit(n_splits=k, test_size=0.25, random_state=RANDOM_SEED)

    # Lista en la que guardamos las perdidas encontradas en cada fold
    losses = []

    # Iteramos usando el split que nos da sklearn
    for train_index, validation_index in ss.split(train_dataset):

        # Tenemos los indices de los elementos, asi que tomamos los dos datasets
        # usando dichos indices
        train_fold = [train_dataset[idx] for idx in train_index]
        validation_fold = [train_dataset[idx] for idx in validation_index]

        # Transformamos los datasets en dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_fold,
            batch_size = ONLINE_BATCH_SIZE,
            shuffle = True,
            num_workers = NUM_WORKERS,
            pin_memory = True,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_fold,
            batch_size = ONLINE_BATCH_SIZE,
            shuffle = True,
            num_workers = NUM_WORKERS,
            pin_memory = True,
        )

        # Entrenamos la red
        _ = train_model_online(
            net = net,
            path = os.path.join(BASE_PATH, "tmp"),
            parameters = parameters,
            train_loader = train_loader,
            validation_loader = validation_loader,
            name = "SiameseNetworkOnline",
            logger = SilentLogger(),
            snapshot_iterations = None
        )

        # Evaluamos la red en el fold de validacion
        net.eval()
        loss = metrics.calculate_mean_triplet_loss_online(net, validation_loader, parameters["criterion"], 1.0)
        loss = float(loss) # Pasamos el tensor de un unico elemento a un float simple

        # Añadimos el loss a nuestra lista
        losses.append(loss)

    # Devolvemos el array en formato numpy para que sea mas comodo trabajar con ella
    return np.array(losses)



# Controlamos si queremos realizar el hyperparameater tuning o no
if SKIP_HYPERPARAMTER_TUNING is False:

    # Para controlar que parametros ya hemos explorado y queremos saltar
    already_explored_parameters = [
        # Embedding dimension, learning rate, margin
        (2, 0.0001, 0.01),
        (2, 0.0001, 1),
        (3, 0.0001),
    ]

    # Parametros que queremos mover
    #margin_values = [0.01, 0.1, 1.0]
    # TODO -- volver a poner todos los valores
    margin_values = [1.0]
    learning_rate_values = [0.0001, 0.001, 0.01]
    embedding_dimension_values = [2, 3, 4]

    # Parametros que fijamos de antemano
    epochs = HYPERPARAMETER_TUNING_EPOCHS

    # Llevamos la cuenta de los mejores parametros y el mejor error encontrados hasta
    # el momento
    best_loss = None
    best_parameters = {
        "embedding_dimension": None,
        "lr": None,
        "margin": None
    }

    # Exploramos las combinaciones de parametros
    for margin in margin_values:
        for learning_rate in learning_rate_values:
            for embedding_dimension in embedding_dimension_values:

                print(f"Optimizando para margin: {margin}, lr: {learning_rate}, embedding_dim: {embedding_dimension}")

                # Comprobamos si tenemos que saltarnos el calculo de algun valor
                # porque ya se haya hecho
                if (embedding_dimension, learning_rate, margin) in already_explored_parameters:
                    print("\tSaltando este calculo porque ya se realizo")
                    continue

                # Definimos el modelo que queremos optimizar
                net = ResNet18(embedding_dimension)

                # En este caso, al no estar trabajando con los minibatches
                # (los usamos directamente como nos los da pytorch), no tenemos
                # que manipular los tensores
                net.set_permute(False)

                parameters = dict()
                parameters["epochs"] = epochs
                parameters["lr"] = learning_rate
                parameters["criterion"] = BatchHardTripletLoss(margin)
                logger = SilentLogger()

                # Usamos nuestra propia funcion de cross validation para validar el modelo
                losses = custom_cross_validation(net, parameters, train_dataset, k = NUMBER_OF_FOLDS)
                print(f"El loss conseguido es {losses.mean()}")
                print("")

                # Comprobamos si hemos mejorado la funcion de perdida
                # En cuyo caso, actualizamos nuestra estructura de datos y, sobre todo, mostramos
                # por pantalla los nuevos mejores parametros
                basic_condition = math.isnan(losses.mean()) is False             # Si es NaN no entramos al if
                enter_condition = best_loss is None or losses.mean() < best_loss # Entramos al if si mejoramos la perdida
                compound_condition = basic_condition and enter_condition
                if compound_condition:

                    # Actualizamos nuestra estructura de datos
                    best_loss = losses.mean()
                    best_parameters = {
                        "embedding_dimension": embedding_dimension,
                        "lr": learning_rate,
                        "margin": margin,
                    }

                    # Mostramos el cambio encontrado
                    print("==> ENCONTRADOS NUEVOS MEJORES PARAMETROS")
                    print(f"Mejores parametros: {best_parameters}")
                    print(f"Mejor loss: {best_loss}")




# # Training of the model

# ## Selecting the network and tweaking some parameters


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


# ## Defining the loggers we want to use


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


# ## Running the training loop


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
                snapshot_iterations = None
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
            snapshot_iterations = None
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


# # Model evaluation

# We start computing the *silhouette* metric for the produced embedding, on train, validation and test set:


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


# # Plot of the embedding
#
# - If the dimension of the embedding is 2, then we can plot how the transformation to a classificator works:


with torch.no_grad():
    classifier.scatter_plot()


# # Evaluating the obtained classifier
#
# - Now that we adapted our network to a classification task, we can compute some classification metrics


with torch.no_grad():
    try_to_clean_memory()
    classifier.embedder.set_permute(False)

    metrics = evaluate_model(classifier, train_loader, test_loader)
    pprint(metrics)

    classifier.embedder.set_permute(True)
