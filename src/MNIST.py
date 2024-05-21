# MNIST
# ==============================================================================

# Global Parameters of the Notebook
# ==============================================================================

import datetime
import gc
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torchvision
import torchvision.transforms as transforms

import wandb
from lib import (embedding_to_classifier, filesystem, loss_functions_blog,
                 metrics, split_dataset, train_loggers, trainers, utils)


@dataclass
class GlobalParameters:
    def __init__(self):
        #  Where are we running the script:
        #      - "local" => our computer
        #      - "remote" => Google Colab
        #      - "ugr" => NGPU UGR
        self.running_env: str = "ugr"

        # TODO -- ADAM's script uses 1 and it takes almost the same time
        self.num_workers = 1

        self.__init_path_params()
        self.__init_ml_params()
        self.__init_loss_params()
        self.__init_hptuning_params()
        self.__init_evaluation_metrics_params()
        self.__init_section_params()
        self.__init_wandb_params()

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

        # Path where we store training / test data
        self.data_path = os.path.join(self.base_path, "data")

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
        self.P: int = 16
        self.K: int = 8

        self.embedding_dimension = 5

        # Minibatches must have size multiple of `P*K` in order to perform P-K sampling
        # So we can use `n * self.P * self.K`
        self.batch_size = self.P * self.K

        # TODO -- previously was 20 training epochs
        self.training_epochs = 20
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.margin = 1.0

        #  self.logging_iterations = self.batch_size * 10
        self.loggin_iterations = 50  # TODO <- Value set by Adam

        # Logging is very slow so just use a small portion of the data
        self.online_logger_train_percentage = 0.005
        self.online_logger_validation_percentage = 0.005

        # TODO -- we are not using this
        # Choose which model we're going to use
        # Can be "ResNet18", "LightModel", "LFWResNet18", "LFWLightModel", "FGLightModel",
        #        "CACDResNet18", "CACDResNet50"
        self.net_model = "LightModel"

        self.normalize_model_output = False

    def __init_loss_params(self):
        # Can be either `"hard"` or `"all"`
        self.loss_batch_variant = "hard"

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
        self.accuracty_at_k_value = 5

    def __init_section_params(self):
        self.skip_hptuning = True

        # When skipping training, we can load the latest cached model and evaluate it
        self.skip_training = False

        self.skip_profiling = True

    def __init_wandb_params(self):
        self.wandb_project_name = "MNIST dataset"
        self.wandb_run_name = str(datetime.datetime.now())

    def dict(self) -> Dict:
        """
        Wandb need a dictionary representation of the class for saving values in
        their system. So this method tries its best to create a dict repr for
        this data class
        """

        return self.__dict__


GLOBALS = GlobalParameters()


# Configuration for Weigths and Biases
# ==============================================================================
#
# - We're going to use `wandb` for tracking the training of the models
# - We use our `GLOBALS` dict to init wandb, that is going to keep track of all
# of that parameters

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


def try_to_clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


# Load the data and use our custom sampler
# ==============================================================================

# Transformations that we want to apply when loading the data
# Now we are only transforming images to tensors (pythorch only works with tensors)
# But we can apply here some normalizations
transform = transforms.Compose(
    [
        transforms.ToTensor()
        # TODO -- apply some normalizations here
    ]
)

# Load the dataset
# torchvision has a method to download and load the dataset
train_dataset = torchvision.datasets.MNIST(
    root=GLOBALS.data_path,
    train=True,
    download=True,
    transform=transform,
)

test_dataset = torchvision.datasets.MNIST(
    root=GLOBALS.data_path,
    train=False,
    download=True,
    transform=transform,
)

# Train -> train / validation split
train_dataset, validation_dataset = split_dataset.split_dataset(train_dataset, 0.8)

# Data loaders to access the datasets
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=GLOBALS.batch_size,
    shuffle=True,
    num_workers=GLOBALS.num_workers,
    pin_memory=True,
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=GLOBALS.batch_size,
    shuffle=True,
    num_workers=GLOBALS.num_workers,
    pin_memory=True,
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=GLOBALS.batch_size,
    shuffle=True,
    num_workers=GLOBALS.num_workers,
    pin_memory=True,
)

# Choose the loss function to use
# ==============================================================================
batch_loss_function = None
if GLOBALS.loss_batch_variant == "hard":
    batch_loss_function = loss_functions_blog.HardTripletLoss(GLOBALS.margin)
if GLOBALS.loss_batch_variant == "all":
    batch_loss_function = loss_functions_blog.BatchAllTtripletLoss(
        GLOBALS.margin,
    )

# Select the network that we are going to use
# ==============================================================================

from adambielski_lib import networks as adamnetworks

cuda = torch.cuda.is_available()
net = adamnetworks.EmbeddingNet()
if cuda:
    net = net.cuda()

#  # TODO -- put them back again
#  ## Defining the loggers we want to use
#  # ==============================================================================

#  print("=> Creating the training loggers that we are going to use")

#  # Define the loggers we want to use
#  triplet_loss_logger = TripletLoggerOnline(
#      net=net,
#      iterations=GLOBALS.loggin_iterations,
#      loss_func=parameters["criterion"],
#      train_percentage=GLOBALS.online_logger_train_percentage,
#      validation_percentage=GLOBALS.online_logger_validation_percentage,
#      greater_than_zero=GLOBALS.use_gt_zero_mean_loss,
#  )

#  cluster_sizes_logger = IntraClusterLogger(
#      net=net,
#      iterations=GLOBALS.logging_iterations,
#      train_percentage=GLOBALS.online_logger_train_percentage,
#      validation_percentage=GLOBALS.online_logger_validation_percentage,
#  )

#  intercluster_metrics_logger = InterClusterLogger(
#      net=net,
#      iterations=GLOBALS.logging_iterations,
#      train_percentage=GLOBALS.online_logger_train_percentage,
#      validation_percentage=GLOBALS.online_logger_validation_percentage,
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

# TODO -- delete this code
logger = train_loggers.CompoundLogger([])


## Running the training loop
# ==============================================================================

parameters = dict()
parameters["epochs"] = GLOBALS.training_epochs
parameters["lr"] = GLOBALS.learning_rate
parameters["weigth_decay"] = GLOBALS.weight_decay

# We use the loss function that depends on the global parameter BATCH_TRIPLET_LOSS_FUNCTION
# We selected this loss func in *Choose the loss function to use* section
parameters["criterion"] = batch_loss_function


ts = time.time()
training_history = trainers.train_model_online(
    net=net,
    path=os.path.join(GLOBALS.base_path, "tmp"),
    parameters=parameters,
    train_loader=train_loader,
    validation_loader=test_loader,
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
        data_loader=train_loader,
        network=net,
        max_examples=len(train_loader),
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
        data_loader=train_loader,
        network=net,
        max_examples=len(train_loader),
        fast_implementation=False,
    )
    test_rank_at_five = metrics.rank_accuracy(
        k=5,
        data_loader=test_loader,
        network=net,
        max_examples=len(test_loader),
        fast_implementation=False,
    )

    print("=> 📈 Final Metrics")
    print(f"Train Rank@1 Accuracy: {train_rank_at_one}")
    print(f"Test Rank@1 Accuracy: {test_rank_at_one}")
    print(f"Train Rank@5 Accuracy: {train_rank_at_five}")
    print(f"Test Rank@5 Accuracy: {test_rank_at_five}")
    print("")

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

    print("=> 📈 Silhouette metrics")
    train_silh = metrics.silhouette(train_loader, net)
    print(f"Silhouette in training loader: {train_silh}")

    test_silh = metrics.silhouette(test_loader, net)
    print(f"Silhouette in test loader: {test_silh}")
    print("")

    # Put this info in wandb
    wandb.log(
        {
            "Final Training silh": train_silh,
            "Final Test silh": test_silh,
        }
    )

    net.set_permute(True)


# Now take the classifier from the embedding and use it to compute some classification metrics:
with torch.no_grad():
    # Try to clean memory, because we can easily run out of memory
    # This provoke the notebook to crash, and all in-memory objects to be lost
    try_to_clean_memory()

    # With hopefully enough memory, try to convert the embedding to a classificator
    number_neigbours = 3
    classifier = embedding_to_classifier.EmbeddingToClassifier(
        net,
        k=number_neigbours,
        data_loader=train_loader,
        embedding_dimension=2,
    )

# See how it works on a small test set
with torch.no_grad():
    net.set_permute(False)

    # Show only `max_iterations` classifications
    counter = 0
    max_iterations = len(test_dataset)

    correct = 0

    for img, img_class in test_dataset:
        predicted_class = classifier.predict(img)

        if img_class == predicted_class[0]:
            correct += 1

        counter += 1
        if counter == max_iterations:
            break

    accuracy = correct / max_iterations
    print(f"=> 📈 Metrics on {max_iterations} test images")
    print(f"Accuracy: {(accuracy * 100):.3f}%")
    print("")

    net.set_permute(True)


# Plot of the embedding
# ==============================================================================
#
# - If the dimension of the embedding is 2, then we can plot how the transformation to a classificator works:
# - That logic is encoded in the `scatter_plot` method
with torch.no_grad():
    print("=> Plotting the embedding that we've learned")
    classifier.scatter_plot(os.path.join(GLOBALS.plots_path, "embedding.png"))
