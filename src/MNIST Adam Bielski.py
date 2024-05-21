# MNIST
# ==============================================================================

# Global Parameters of the Notebook
# ==============================================================================

import datetime
import gc
import os
from dataclasses import dataclass
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import wandb
from lib import embedding_to_classifier, metrics, utils


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
        # TODO -- we are not using this in ADAM's pipeline!
        self.P: int = 8
        self.K: int = 4

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


# Import MNIST dataset
# ==============================================================================


# TODO -- values from ADAM's script
mean, std = 0.1307, 0.3081

print("=> Downloading the MNIST dataset")
train_dataset = torchvision.datasets.MNIST(
    GLOBALS.data_path,
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    ),
)
test_dataset = torchvision.datasets.MNIST(
    GLOBALS.data_path,
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
# TODO -- ADAM -- understand what n_classes and n_samples mean in this context
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = adamdatasets.BalancedBatchSampler(
    train_dataset.train_labels, n_classes=10, n_samples=25
)
test_batch_sampler = adamdatasets.BalancedBatchSampler(
    test_dataset.test_labels, n_classes=10, n_samples=25
)

cuda = torch.cuda.is_available()
kwargs = {"num_workers": GLOBALS.num_workers, "pin_memory": True} if cuda else {}
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

batch_loss_function = adamlosses.OnlineTripletLoss(
    GLOBALS.margin, adamutils.RandomNegativeTripletSelector(GLOBALS.margin)
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

optimizer = torch.optim.Adam(
    net.parameters(), lr=GLOBALS.learning_rate, weight_decay=GLOBALS.weight_decay
)

# TODO -- ADAM -- What is this?
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

adamtrainer.fit(
    online_train_loader,
    online_test_loader,
    net,
    batch_loss_function,
    optimizer,
    scheduler,
    GLOBALS.training_epochs,
    cuda,
    GLOBALS.loggin_iterations,
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


def plot_embeddings(embeddings, targets, title: str, xlim=None, ylim=None):
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
    try:
        plt.savefig(os.path.join(GLOBALS.plots_path, title))
    except Exception as e:
        print("Could not save figure in disk")
        print(f"Reason was: {e=}")


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
plot_embeddings(train_embeddings_otl, train_labels_otl, title="Train embeddings")
val_embeddings_otl, val_labels_otl = extract_embeddings(online_test_loader, net)
plot_embeddings(val_embeddings_otl, val_labels_otl, title="Validation embeddings")

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


# Model evaluation
# ==============================================================================

# Use the network to perform a retrieval task and compute rank@1 and rank@5 accuracy
with torch.no_grad():
    net.set_permute(False)

    train_rank_at_one = metrics.rank_accuracy(
        k=1,
        data_loader=online_train_loader,
        network=net,
        max_examples=len(online_train_loader),
        fast_implementation=False,
    )
    test_rank_at_one = metrics.rank_accuracy(
        k=1,
        data_loader=online_test_loader,
        network=net,
        max_examples=len(online_test_loader),
        fast_implementation=False,
    )
    train_rank_at_five = metrics.rank_accuracy(
        k=5,
        data_loader=online_train_loader,
        network=net,
        max_examples=len(online_train_loader),
        fast_implementation=False,
    )
    test_rank_at_five = metrics.rank_accuracy(
        k=5,
        data_loader=online_test_loader,
        network=net,
        max_examples=len(online_test_loader),
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
    train_silh = metrics.silhouette(online_train_loader, net)
    print(f"Silhouette in training loader: {train_silh}")

    test_silh = metrics.silhouette(online_test_loader, net)
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
        data_loader=online_train_loader,
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
