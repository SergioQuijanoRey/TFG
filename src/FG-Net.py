# FG-NET
# ==============================================================================
#
# - Check `FG-Net Notebook.ipynb` for:
#  - Some notes on other papers related to our work
#  - EDA of the dataset

# Global Parameters of the Notebook
# ==============================================================================

import datetime
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class GlobalParameters:
    """Dataclass that will hold all the parameters for this experiment"""

    def __init__(self):
        #  Where are we running the script:
        #      - "local" => our computer
        #      - "remote" => Google Colab
        #      - "ugr" => NGPU UGR
        self.running_env: str = "ugr"

        self.num_workers = 1
        self.random_seed = 123456789

        self.__init_path_params()
        self.__init_ml_params()
        self.__init_loss_params()
        self.__init_hptuning_params()
        self.__init_evaluation_metrics_params()
        self.__init_section_params()
        self.__init_wandb_params()
        self.__init_knn_params()
        self.__init_data_augmentation_params()

    def __init_path_params(self):
        # Select the base path, which depends on the running env
        self.base_path: str = ""
        if self.running_env == "local":
            self.base_path = "./"
        elif self.running_env == "remote":
            self.base_path = "/content/drive/MyDrive/Colab Notebooks/"
        elif self.running_env == "ugr":
            self.base_path = "/mnt/homeGPU/squijano/TFG/"
        else:
            raise ValueError(
                f"Running env '{self.running_env}' is not a valid running env"
            )

        # Path to our lib dir
        self.lib_path = os.path.join(self.base_path, "lib")

        # Path where we store the two datasets
        self.data_path = os.path.join(self.base_path, "data")
        self.cacd_data_path = os.path.join(self.data_path, "CACD")

        # And where the images are stored
        self.image_dir_path = os.path.join(self.data_path, "FGNET/images")
        self.cacd_image_dir_path = os.path.join(self.cacd_data_path, "CACD2000")

        # URLs from where we dowload the data
        self.dataset_url = "http://yanweifu.github.io/FG_NET_data/FGNET.zip"
        self.cacd_dataset_url = (
            "https://drive.google.com/file/d/1hYIZadxcPG27Fo7mQln0Ey7uqw1DoBvM/view"
        )

        # Path where we can store figures
        self.plots_path = os.path.join(self.base_path, "plots")

        # Dir with all cached models
        # This cached models can be loaded from disk when training is skipped
        self.model_cache_folder = os.path.join(self.base_path, "cached_models")

        # Cache for the augmented dataset
        self.augmented_dataset_cache_file = os.path.join(
            self.base_path, "cached_augmented_dataset.pt"
        )

        # Binary file where the stats of the profiling are saved
        self.profile_save_file = os.path.join(self.base_path, "training_profile.stat")

        # The SQLITE database where we are going to store the hp tuning process
        self.optuna_database = f"sqlite:///{self.base_path}/hp_tuning_optuna.db"

    def __init_ml_params(self):
        # P-K sampling main parameters
        self.P: int = 8
        self.K: int = 4

        # Minibatches must have size multiple of `P*K` in order to perform P-K sampling
        # So we can use `n * self.P * self.K`
        self.batch_size = self.P * self.K

        # TODO -- originaly was 9
        self.embedding_dimension = 5

        self.training_epochs = 1
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.margin = 0.840

        # How many elements we see before logging. We use `P*K` so we align with
        # our batch size
        self.logging_iterations = self.P * self.K * 1_000

        # Logging is very slow so just use a small portion of the data
        self.online_logger_train_percentage = 0.005
        self.online_logger_validation_percentage = 0.005

        # Choose which model we're going to use
        # Can be "ResNet18", "LightModel", "LFWResNet18", "LFWLightModel", "FGLightModel",
        #        "CACDResNet18", "CACDResNet50"
        self.net_model = "CACDResNet18"

        self.normalized_model_output = False

    def __init_loss_params(self):
        # Can be either `"hard"` or `"all"`
        self.batch_triplet_loss_function = "hard"

        # Whether or not use softplus dist function instead of vanilla euclidean dist
        self.use_softplus = False

        # Count all summands in the mean loss or only those summands greater than zero
        self.use_gt_zero_mean_loss = True

        # Wether or not add penalty term to the loss function, and how hardly
        self.add_norm_penalty = False
        self.penalty_factor = 0.6

        # Wether or not apply gradient clipping to avoid some stability issues
        self.gradient_clipping: Optional[float] = None

    def __init_hptuning_params(self):
        # Epochs used in k-Fold Cross validation
        self.hptuning_epochs = 1

        # Number of tries in the optimization process
        self.hptuning_tries = 300

        self.hptuning_kfolds = 2

        # Wether to use the validation set in the hp tuning process or to use k-fold
        # cross validation (which is more robust but way slower)
        self.hptuning_fast = True

    def __init_evaluation_metrics_params(self):
        # Number of candidates that we are going to consider in the retrieval task,
        # used in the Rank@K accuracy metric
        # We use k = 1 and k = this value
        self.accuracy_at_k_value = 5

    def __init_section_params(self):
        self.skip_hptuning = True

        # When skipping training, we can load the latest cached model and evaluate it
        self.skip_training = False
        self.use_cached_model = False

        self.skip_profiling = True

        # Wether or not skip dataset augmentation and use the cached augmented dataset
        self.use_cached_augmented_dataset = False

    def __init_wandb_params(self):
        self.wandb_project_name = "FG-NET dataset"
        self.wandb_run_name = str(datetime.datetime.now())

    def __init_knn_params(self):
        self.number_neighbours = 4

    def __init_data_augmentation_params(self):
        self.lazy_data_augmentation = True
        self.avoid_custom_sampler_fail = True

        self.image_shape = (200, 200)
        self.rotate_augm_degrees = (0, 20)

    def dict(self) -> Dict:
        """
        Wandb need a dictionary representation of the class for saving values in
        their system. So this method tries its best to create a dict repr for
        this data class
        """

        return self.__dict__


GLOBALS = GlobalParameters()

# Importing the modules we are going to use
# ==============================================================================

import copy
import cProfile
import enum
import functools
import gc
import logging
import math
import time
from collections import Counter
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
from lib.loss_functions import (
    AddSmallEmbeddingPenalization,
    BatchAllTripletLoss,
    BatchHardTripletLoss,
    MeanTripletBatchTripletLoss,
)
from lib.models import *
from lib.models import (
    CACDResnet18,
    CACDResnet50,
    FGLigthModel,
    LFWLightModel,
    LFWResNet18,
    NormalizedNet,
    ResNet18,
    RetrievalAdapter,
)
from lib.sampler import CustomSampler
from lib.train_loggers import (
    CompoundLogger,
    InterClusterLogger,
    IntraClusterLogger,
    LocalRankAtKLogger,
    RankAtKLogger,
    SilentLogger,
    TrainLogger,
    TripletLoggerOffline,
    TripletLoggerOnline,
)
from lib.trainers import train_model_online
from lib.visualizations import *

# Server security check
# ==============================================================================
#
# - Sometimes UGR's server does not provide GPU access
# - In that case, fail fast so we start ASAP debugging the problem

if GLOBALS.running_env == "ugr" and torch.cuda.is_available() is False:
    raise Exception(
        "`torch.cuda.is_available()` returned false, so we dont have access to GPU's"
    )


# TODO -- DELETE
torch.autograd.set_detect_anomaly(True)

# Configuration for Weigths and Biases
# ==============================================================================
#
# - We're going to use `wandb` for tracking the training of the models
# - We use our `GLOBALS` methot to represent it as a dict to init wandb, that is
# going to keep track of all of that parameters

# If we're running in UGR's servers, we need to set some ENV vars
# Otherwise, wandb is going to write to dirs that it has no access
# Also, pytorch tries to save pretrained models in the home folder
if GLOBALS.running_env == "ugr":
    print("-> Changing dir env values")
    utils.change_dir_env_vars(base_path=GLOBALS.base_path)
    print("-> Changing done!")
    print("")

    print("-> Login again to WANDB")
    utils.login_wandb()
    print("-> Login done!")
    print("")

# Init the wandb tracker
# We need to do this before `wandb.login`
wandb.init(
    project=GLOBALS.wandb_project_name,
    name=GLOBALS.wandb_run_name,
    config=GLOBALS.dict(),
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


# Importing and preparing the data
# ==============================================================================

## Dataset downloading
# ==============================================================================
#
# - There is no package in pytorch / torchvision for this dataset
#  - So we download the dataset from an URL

datasets.download_fg_dataset(
    GLOBALS.data_path, GLOBALS.dataset_url, can_skip_download=True
)

datasets.download_cacd_dataset(
    GLOBALS.cacd_data_path,
    GLOBALS.cacd_dataset_url,
    can_skip_download=True,
    can_skip_extraction=True,
)

## Dataset loading
# ==============================================================================
#
# As mentioned before, we have to use our custom implementation for pytorch
# `Dataset` class


# Transformations that we want to apply when loading the data
# TODO -- we are using the same transform for both datasets
transform = transforms.Compose(
    [
        # First, convert to a PIL image so we can resize
        transforms.ToPILImage(),
        # Some images are colored, other images are black and white
        # So convert all the images to black and white, but having three channels
        transforms.Grayscale(num_output_channels=3),
        # Images have different shapes, so this normalization is needed
        transforms.Resize(GLOBALS.image_shape, antialias=True),
        # Pytorch only work with tensors
        transforms.ToTensor(),
        # Some normalization
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

print("=> Wrapping the raw data into a `FGDataset")
fgnet_dataset = datasets.FGDataset(path=GLOBALS.image_dir_path, transform=transform)

print("=> Wrapping the raw data into `CACDDataset`")
cacd_dataset = datasets.CACDDataset(
    path=GLOBALS.cacd_image_dir_path, transform=transform
)

## Splitting the dataset
# ==============================================================================

# This function returns a `WrappedSubset` so we can still have access to
# `targets` attribute. With `Subset` pytorch class we cannot do that
# This split function returns subsets with disjoint classes. That is to say,
# if there is one person in one dataset, that person cannot appear in the
# other datset. Thus, percentages may vary a little
print("=> Splitting the dataset")
train_dataset, validation_dataset = split_dataset.split_dataset_disjoint_classes(
    cacd_dataset, 0.8
)
test_dataset = fgnet_dataset

print("--> Dataset sizes:")
print(f"\tTrain dataset: {len(train_dataset) / len(cacd_dataset) * 100}%")
print(f"\tValidation dataset: {len(validation_dataset) / len(cacd_dataset) * 100}%")
print("")

print("--> Logging sizes:")
print(f"\tTrain dataset: {len(train_dataset) * GLOBALS.online_logger_train_percentage}")
print(
    f"\tValidation dataset: {len(validation_dataset) * GLOBALS.online_logger_validation_percentage}"
)
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
    batch_size=GLOBALS.batch_size,
    sampler=CustomSampler(
        GLOBALS.P,
        GLOBALS.K,
        train_dataset,
        avoid_failing=GLOBALS.avoid_custom_sampler_fail,
    ),
)

# TODO -- here I don't know if use default sampler or custom sampler
validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=GLOBALS.batch_size,
    sampler=CustomSampler(
        GLOBALS.P,
        GLOBALS.K,
        dataset=validation_dataset,
        avoid_failing=GLOBALS.avoid_custom_sampler_fail,
    ),
)

# TODO -- here I don't know if use default sampler or custom sampler
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=GLOBALS.batch_size,
    shuffle=True,
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
if GLOBALS.use_cached_augmented_dataset is True:
    train_dataset_augmented = torch.load(GLOBALS.augmented_dataset_cache_file)

# We have to do the data augmentation if we mark that we want to do it (section parameter)
# Or if the cached dataset was done for other number of images (ie. for 4 when
# now we want 32)
if (
    GLOBALS.use_cached_augmented_dataset == False
    or train_dataset_augmented.min_number_of_images != GLOBALS.K
):
    print("=> Performing data augmentation")

    # Select the data augmentation mechanism
    AugmentationClass = (
        LazyAugmentatedDataset
        if GLOBALS.lazy_data_augmentation is True
        else AugmentatedDataset
    )

    train_dataset_augmented = AugmentationClass(
        base_dataset=train_dataset,
        min_number_of_images=GLOBALS.K,
        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        transform=transforms.Compose(
            [
                # NOTE: We have normalized our images to be (3, 300, 300), so new
                # randomly generated images have to have the same shape
                transforms.RandomResizedCrop(size=GLOBALS.image_shape, antialias=True),
                transforms.RandomRotation(degrees=GLOBALS.rotate_augm_degrees),
                transforms.RandomAutocontrast(),
            ]
        ),
    )

    validation_dataset_augmented = AugmentationClass(
        base_dataset=validation_dataset,
        min_number_of_images=GLOBALS.K,
        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        transform=transforms.Compose(
            [
                # NOTE: We have normalized our images to be (3, 300, 300), so new
                # randomly generated images have to have the same shape
                transforms.RandomResizedCrop(size=GLOBALS.image_shape, antialias=True),
                transforms.RandomRotation(degrees=GLOBALS.rotate_augm_degrees),
                transforms.RandomAutocontrast(),
            ]
        ),
    )

    # TODO -- augmented dataset also for validation, in case we think that the
    #         CustomSampler is a good idea for the validation dataset
    # Save the augmented dataset to cache
    torch.save(train_dataset_augmented, GLOBALS.augmented_dataset_cache_file)

# Now put a loader in front of the augmented dataset
train_loader_augmented = torch.utils.data.DataLoader(
    train_dataset_augmented,
    batch_size=GLOBALS.batch_size,
    sampler=CustomSampler(
        GLOBALS.P,
        GLOBALS.K,
        train_dataset_augmented,
        avoid_failing=GLOBALS.avoid_custom_sampler_fail,
    ),
)

validation_loader_augmented = torch.utils.data.DataLoader(
    validation_dataset_augmented,
    batch_size=GLOBALS.batch_size,
    sampler=CustomSampler(
        GLOBALS.P,
        GLOBALS.K,
        validation_dataset_augmented,
        avoid_failing=GLOBALS.avoid_custom_sampler_fail,
    ),
)


## Remove previous datasets
# ==============================================================================
#
# - If we're not doing hyperparameter tuning, we don't need to hold previous dataset and dataloader


if GLOBALS.skip_hptuning is True:
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
if GLOBALS.batch_triplet_loss_function == "hard":
    batch_loss_function = BatchHardTripletLoss(
        GLOBALS.margin,
        use_softplus=GLOBALS.use_softplus,
        use_gt_than_zero_mean=GLOBALS.use_gt_zero_mean_loss,
    )
if GLOBALS.batch_triplet_loss_function == "all":
    batch_loss_function = BatchAllTripletLoss(
        GLOBALS.margin,
        use_softplus=GLOBALS.use_softplus,
        use_gt_than_zero_mean=GLOBALS.use_gt_zero_mean_loss,
    )

# Sanity check
if batch_loss_function is None:
    raise Exception(
        f"BATCH_TRIPLET_LOSS global parameter got unexpected value: {GLOBALS.batch_triplet_loss_function}"
    )


# Choose wheter to add embedding norm or not
# ==============================================================================


if GLOBALS.add_norm_penalty:
    batch_loss_function = AddSmallEmbeddingPenalization(
        base_loss=batch_loss_function,
        penalty_factor=GLOBALS.penalty_factor,
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
        k=1,
        data_loader=validation_fold,
        network=net,
        max_examples=len(validation_fold),
        fast_implementation=False,
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
        "Network", ["CACDResNet18", "CACDResNet50", "FGLightModel"]
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
        gradient_clipping = trial.suggest_float(
            "Gradient Clipping Value", 0.00001, 10.0
        )

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
        base_dataset=train_dataset,
        min_number_of_images=k,
        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        # Again, we want to end with the normalized shape
        transform=transforms.Compose(
            [
                transforms.RandomResizedCrop(size=GLOBALS.image_shape, antialias=True),
                transforms.RandomRotation(degrees=GLOBALS.rotate_augm_degrees),
                transforms.RandomAutocontrast(),
            ]
        ),
    )

    # Put some dataloaders
    # TODO -- esto en la implementacion original se hace en el `loader_generator`
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=p * k,
        sampler=CustomSampler(
            p,
            k,
            train_dataset,
            avoid_failing=GLOBALS.avoid_custom_sampler_fail,
        ),
    )

    # And with p, k values we can define the way we use the laoder generator
    # This p, k values are captured in the outer scope for the `CustomSampler`
    def loader_generator(
        fold_dataset: split_dataset.WrappedSubset, fold_type: hptuning.FoldType
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
        if (
            fold_type is hptuning.FoldType.TRAIN_FOLD
            or fold_type is hptuning.FoldType.VALIDATION_FOLD
        ):
            fold_dataset_augmented = LazyAugmentatedDataset(
                base_dataset=fold_dataset,
                min_number_of_images=k,
                # Remember that the trasformation has to be random type
                # Otherwise, we could end with a lot of repeated images
                transform=transforms.Compose(
                    [
                        transforms.RandomResizedCrop(
                            size=GLOBALS.image_shape, antialias=True
                        ),
                        transforms.RandomRotation(degrees=GLOBALS.rotate_augm_degrees),
                        transforms.RandomAutocontrast(),
                    ]
                ),
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
                batch_size=p * k,
                sampler=CustomSampler(
                    p,
                    k,
                    fold_dataset,
                    avoid_failing=GLOBALS.avoid_custom_sampler_fail,
                ),
            )
        elif fold_type is hptuning.FoldType.VALIDATION_FOLD:
            loader = torch.utils.data.DataLoader(
                fold_dataset_augmented,
                batch_size=p * k,
            )
        else:
            raise ValueError(
                f"{fold_type=} enum value is not managed in if elif construct!"
            )

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
    def network_trainer(
        fold_dataloader: DataLoader, net: torch.nn.Module
    ) -> torch.nn.Module:
        parameters = dict()
        parameters["epochs"] = GLOBALS.hptuning_epochs
        parameters["lr"] = learning_rate
        parameters["criterion"] = BatchHardTripletLoss(
            margin, use_softplus=softplus, use_gt_than_zero_mean=True
        )

        # Wether or not use norm penalization
        if use_norm_penalty:
            parameters["criterion"] = AddSmallEmbeddingPenalization(
                base_loss=parameters["criterion"],
                penalty_factor=norm_penalty,
            )

        _ = train_model_online(
            net=net,
            path=os.path.join(GLOBALS.base_path, "tmp_hp_tuning"),
            parameters=parameters,
            train_loader=fold_dataloader,
            validation_loader=None,
            name="Hyperparameter Tuning Network",
            logger=SilentLogger(),
            snapshot_iterations=None,
            gradient_clipping=gradient_clipping,
            fail_fast=True,
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
                train_dataset=train_dataset_augmented,
                k=GLOBALS.hptuning_kfolds,
                random_seed=GLOBALS.random_seed,
                network_creator=network_creator,
                network_trainer=network_trainer,
                loader_generator=loader_generator,
                loss_function=loss_function,
            )
            print(f"Array of losses: {losses=}")
            print(f"Obtained loss (cross validation mean) is {losses.mean()=}")

        except Exception as e:
            # Show that cross validation failed for this combination
            msg = "Could not run succesfully k-fold cross validation for this combination of parameters\n"
            msg = msg + f"Error was: {e}\n"
            print(msg)

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
if GLOBALS.skip_hptuning is False:
    # We want to chose the `objective` implementation to use. But optuna only
    # accepts functions with the shape `objective(trial)` so get a partial
    # function with the parameter `implementation chosen`
    strat = TuningStrat.HOLDOUT if GLOBALS.hptuning_fast is True else TuningStrat.KFOLD
    partial_objective = lambda trial: objective(trial, implementation=strat)

    print(f"🔎 Started hyperparameter tuning with {strat=}")
    print("")

    study = optuna.create_study(
        direction="maximize",
        study_name="Rank@1 optimization",
        storage=GLOBALS.optuna_database,
        load_if_exists=True,
    )
    study.optimize(partial_objective, n_trials=GLOBALS.hptuning_tries)

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
if GLOBALS.net_model == "ResNet18":
    net = ResNet18(GLOBALS.embedding_dimension)
elif GLOBALS.net_model == "LightModel":
    net = LightModel(GLOBALS.embedding_dimension)
elif GLOBALS.net_model == "LFWResNet18":
    net = LFWResNet18(GLOBALS.embedding_dimension)
elif GLOBALS.net_model == "LFWLightModel":
    net = LFWLightModel(GLOBALS.embedding_dimension)
elif GLOBALS.net_model == "FGLightModel":
    net = FGLigthModel(GLOBALS.embedding_dimension)
elif GLOBALS.net_model == "CACDResNet18":
    net = CACDResnet18(GLOBALS.embedding_dimension)
elif GLOBALS.net_model == "CACDResNet50":
    net = CACDResnet50(GLOBALS.embedding_dimension)
else:
    raise Exception("Parameter 'NET_MODEL' has not a valid value")

# Wrap the model if we want to normalize the output
if GLOBALS.normalized_model_output is True:
    net = NormalizedNet(net)

# The custom sampler takes care of minibatch management
# Thus, we don't have to make manipulations on them
net.set_permute(False)

# Training parameters
parameters = dict()
parameters["epochs"] = GLOBALS.training_epochs
parameters["lr"] = GLOBALS.learning_rate

# We use the loss function that depends on the global parameter BATCH_TRIPLET_LOSS_FUNCTION
# We selected this loss func in *Choose the loss function to use* section
parameters["criterion"] = batch_loss_function

print(net)


## Defining the loggers we want to use
# ==============================================================================


print("=> Creating the training loggers that we are going to use")

# Define the loggers we want to use
triplet_loss_logger = TripletLoggerOnline(
    net=net,
    iterations=GLOBALS.logging_iterations,
    loss_func=parameters["criterion"],
    train_percentage=GLOBALS.online_logger_train_percentage,
    validation_percentage=GLOBALS.online_logger_validation_percentage,
    greater_than_zero=GLOBALS.use_gt_zero_mean_loss,
)

cluster_sizes_logger = IntraClusterLogger(
    net=net,
    iterations=GLOBALS.logging_iterations,
    train_percentage=GLOBALS.online_logger_train_percentage,
    validation_percentage=GLOBALS.online_logger_validation_percentage,
)

intercluster_metrics_logger = InterClusterLogger(
    net=net,
    iterations=GLOBALS.logging_iterations,
    train_percentage=GLOBALS.online_logger_train_percentage,
    validation_percentage=GLOBALS.online_logger_validation_percentage,
)

rank_at_one_logger = RankAtKLogger(
    net=net,
    iterations=GLOBALS.logging_iterations,
    train_percentage=GLOBALS.online_logger_train_percentage,
    validation_percentage=GLOBALS.online_logger_validation_percentage,
    k=1,
)

rank_at_k_logger = RankAtKLogger(
    net=net,
    iterations=GLOBALS.logging_iterations,
    train_percentage=GLOBALS.online_logger_train_percentage,
    validation_percentage=GLOBALS.online_logger_validation_percentage,
    k=GLOBALS.accuracy_at_k_value,
)


local_rank_at_one_logger = LocalRankAtKLogger(
    net=net,
    iterations=GLOBALS.logging_iterations,
    train_percentage=GLOBALS.online_logger_train_percentage,
    validation_percentage=GLOBALS.online_logger_validation_percentage,
    k=1,
)

local_rank_at_k_logger = LocalRankAtKLogger(
    net=net,
    iterations=GLOBALS.logging_iterations,
    train_percentage=GLOBALS.online_logger_train_percentage,
    validation_percentage=GLOBALS.online_logger_validation_percentage,
    k=GLOBALS.accuracy_at_k_value,
)

# Combine them in a single logger
logger = CompoundLogger(
    [
        triplet_loss_logger,
        cluster_sizes_logger,
        intercluster_metrics_logger,
        rank_at_one_logger,
        rank_at_k_logger,
        local_rank_at_one_logger,
        local_rank_at_k_logger,
    ]
)


## Running the training loop
# ==============================================================================

# Check if we want to skip training
if GLOBALS.use_cached_model is False:
    # To measure the time it takes to train
    ts = time.time()

    # Run the training with the profiling
    if GLOBALS.skip_profiling is False:
        _ = cProfile.run(
            """train_model_online(
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
            GLOBALS.profile_save_file,
        )

    # Run the training without the profiling
    else:
        training_history = train_model_online(
            net=net,
            path=os.path.join(GLOBALS.base_path, "tmp"),
            parameters=parameters,
            train_loader=train_loader_augmented,
            validation_loader=validation_loader_augmented,
            name=GLOBALS.net_model,
            logger=logger,
            snapshot_iterations=None,
            gradient_clipping=GLOBALS.gradient_clipping,
        )

    # Compute how long it took
    te = time.time()
    print(f"It took {te - ts} seconds to train")

    # Update the model cache
    filesystem.save_model(net, GLOBALS.model_cache_folder, "online_model_cached")

# In case we skipped training, load the model from cache
else:
    # Choose the function to construct the new network
    if GLOBALS.net_model == "ResNet18":
        net_func = lambda: ResNet18(GLOBALS.embedding_dimension)
    elif GLOBALS.net_model == "LightModel":
        net_func = lambda: LightModel(GLOBALS.embedding_dimension)
    elif GLOBALS.net_model == "LFWResNet18":
        net_func = lambda: LFWResNet18(GLOBALS.embedding_dimension)
    elif GLOBALS.net_model == "LFWLightModel":
        net_func = lambda: LFWLightModel(GLOBALS.embedding_dimension)
    elif GLOBALS.net_model == "FGLightModel":
        net_func = lambda: FGLigthModel(GLOBALS.embedding_dimension)
    elif GLOBALS.net_model == "CADResNet18":
        net_func = lambda: CACDResnet18(GLOBALS.embedding_dimension)
    elif GLOBALS.net_model == "CACDResNet50":
        net_func = lambda: CACDResnet50(GLOBALS.embedding_dimension)
    else:
        raise Exception("Parameter 'NET_MODEL' has not a valid value")

    # Load the model from cache
    net = filesystem.load_model(
        os.path.join(GLOBALS.model_cache_folder, "online_model_cached"), net_func
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
        k=1,
        data_loader=train_loader_augmented,
        network=net,
        max_examples=len(train_loader_augmented),
        fast_implementation=False,
    )
    test_rank_at_one = metrics.rank_accuracy(
        k=1,
        data_loader=test_loader,
        network=net,
        max_examples=len(test_loader),
        fast_implementation=False,
    )
    train_rank_at_five = metrics.rank_accuracy(
        k=5,
        data_loader=train_loader_augmented,
        network=net,
        max_examples=len(train_loader_augmented),
        fast_implementation=False,
    )
    test_rank_at_five = metrics.rank_accuracy(
        k=5,
        data_loader=test_loader,
        network=net,
        max_examples=len(test_loader),
        fast_implementation=False,
    )

    print(f"Train Rank@1 Accuracy: {train_rank_at_one}")
    print(f"Test Rank@1 Accuracy: {test_rank_at_one}")
    print(f"Train Rank@5 Accuracy: {train_rank_at_five}")
    print(f"Test Rank@5 Accuracy: {test_rank_at_five}")

    # Put this info in wandb
    wandb.log(
        {
            "Final Train Rank@1 Accuracy": train_rank_at_one,
            "Final Test Rank@1 Accuracy": test_rank_at_one,
            "Final Train Rank@5 Accuracy": train_rank_at_five,
            "Final Test Rank@5 Accuracy": test_rank_at_five,
        }
    )

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
    wandb.log(
        {
            "Final Training silh": train_silh,
            "Final Validation silh": validation_silh,
            "Final Test silh": test_silh,
        }
    )

    net.set_permute(True)


# Show the "criterion" metric on test set
with torch.no_grad():
    net.set_permute(False)

    core.test_model_online(net, test_loader, parameters["criterion"], online=True)

    net.set_permute(True)


# Now take the classifier from the embedding and use it to compute some classification metrics:
with torch.no_grad():
    # Try to clean memory, because we can easily run out of memory
    # This provoke the notebook to crash, and all in-memory objects to be lost
    try_to_clean_memory()

    # With hopefully enough memory, try to convert the embedding to a classificator
    classifier = EmbeddingToClassifier(
        net,
        k=GLOBALS.number_neighbours,
        data_loader=train_loader_augmented,
        embedding_dimension=GLOBALS.embedding_dimension,
    )

# See how it works on a small test set
with torch.no_grad():
    net.set_permute(False)

    # Show only `max_iterations` classifications
    counter = 0
    max_iterations = 20

    for img, img_class in test_dataset:
        predicted_class = classifier.predict(img)
        print(
            f"True label: {img_class}, predicted label: {predicted_class[0]}, correct: {img_class == predicted_class[0]}"
        )

        counter += 1
        if counter == max_iterations:
            break

    net.set_permute(True)


# Plot of the embedding
# ==============================================================================
#
# - If the dimension of the embedding is 2, then we can plot how the transformation to a classificator works:
# - That logic is encoded in the `scatter_plot` method
with torch.no_grad():
    classifier.scatter_plot()
