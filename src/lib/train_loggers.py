from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import wandb
from typing import List, Tuple, Callable, Dict

import logging
file_logger = logging.getLogger("MAIN_LOGGER")

import src.lib.core as core
import src.lib.board as board
import src.lib.metrics as metrics


class TrainLogger(ABC):
    """
    TrainLogger logs data from the trainning process

    This abstract class acts as an interface for classes that want to be used as loggers for
    the training process
    """

    @abstractmethod
    def log_process(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        epoch_iteration: int
    ) -> Dict[str, float]:
        """
        Logs an iteration of training process. This log can be just printing to terminal or saving
        scalars to a tensorboard

        @param train_loader: dataloader for training data
        @param validation_loader: dataloader for validation data
        @param loss: computed loss in training function
        @param epoch: the epoch where we are at the moment
        @param epoch_iteration: how many single elements have been seen in this epoch

        @returns metrics: a dict containing "name of the metric" -> value of the metric
        """
        pass

    @abstractmethod
    def should_log(self, iteration: int) -> bool:
        """
        Decides wether or not we should log data in this training iteration

        Iterations refer to how many SINGLE ELEMENTS we have seen
        It DOES NOT refer to:
            - How many batches we have seen
            - How many epochs the net has been trained

        @param iteration should be a multiple of the batch size. Otherwise can happen that we
               never log
        """
        pass

class SilentLogger(TrainLogger):
    """Logger that does not log data"""

    def log_process(self, train_loader: DataLoader, validation_loader: DataLoader, epoch: int, iteration: int) -> None:
        # This code should never be triggered, because of `should_log`, but just in case just pass
        pass

    def should_log(self, epoch_iteration: int) -> bool:
        # Always return false in order to never log
        return False

# TODO -- TEST -- write some unit tests for this class
class CompoundLogger(TrainLogger):
    """
    Class that takes a list of loggers, and compount them in one single logger

    This logger calls should_log on each base logger. If one logger should log, it will log with
    its `log_process` method
    """

    def __init__(self, loggers: List[TrainLogger]):
        self.loggers = loggers

        # When a logger wants to log at certain iteration, `CompoundLogger.log_process` has to call
        # that method on that particular logger. So we need this attribute to keep track of which
        # loggers want to log at this iteration
        self.logger_should_log: List[bool] = None

    def log_process(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        epoch_iteration: int
    ) -> Dict[str, float]:

        # TODO -- this break the return type that we've annotated
        ret_values: Dict[str, float] = dict()

        # Iterate over all the loggers that want to log
        for indx, logger in enumerate(self.loggers):
            if self.logger_should_log[indx] is True:

                # Get the returned dict of this logger
                ret_val = logger.log_process(
                    train_loader,
                    validation_loader,
                    epoch,
                    epoch_iteration
                )

                # Check that there is no repeated key
                for key in ret_val.keys():
                    if key in ret_values.keys():

                        # Log some extra information before raising the exception
                        file_logger.error((f"Two loggers return a value with the same key, {key}"))
                        file_logger.error((f"key was: {key}"))
                        file_logger.error((f"ret_values.keys() = {ret_values.keys()}"))

                        raise Exception(f"Two loggers return a value with the same key, {key}")

                # Merge the dict returned by the current logger to our global dict
                # Google Colab does not support expression `ret_values | ret_val`
                ret_values = {**ret_values, **ret_val}

        return ret_values

    def should_log(self, iteration: int) -> bool:

        # Compute the list of loggers that want to log
        self.logger_should_log = [logger.should_log(iteration) for logger in self.loggers]

        # Now, we can tell if there's at least one logger that wants to log
        return any(self.logger_should_log)

# TODO -- remove this class because we're not using it
class ClassificationLogger(TrainLogger):
    """
    Logger for a classifaction problem
    """

    def __init__(self, net: nn.Module, iterations, loss_func, training_perc: float = 1.0, validation_perc: float = 1.0):
        """
        Initializes the logger

        Parameters:
        ===========
        net: the net we are testing
        iterations: how many iterations we have to wait to log again
        loss_func: the loss func we are using to train
        training_perc: the percentage of the training set we are going to use to compute metrics
        validation_perc: the percentage of the validation set we are going to use to compute metrics
        tensorboardwriter: writer to tensorboard logs
        """
        self.iterations = iterations
        self.loss_func = loss_func
        self.net = net
        self.training_perc = training_perc
        self.validation_perc = validation_perc

        # Tensorboard named with the date/time stamp
        self.name = core.get_datetime_str()
        self.tensorboardwriter = board.get_writer(name = self.name)

    def log_process(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        epoch_iteration: int
    ) -> Dict[str, float]:

        # For more performance
        with torch.no_grad():

            # Even more performance
            self.net.eval()

            # Trainning loss and accuracy
            max_examples = int(len(train_loader.dataset) * self.training_perc)
            mean_train_loss = metrics.calculate_mean_loss(self.net, train_loader, max_examples, self.loss_func)
            mean_train_acc = metrics.calculate_accuracy(self.net, train_loader, max_examples)

            # Validation loss and accuracy
            max_examples = int(len(validation_loader.dataset) * self.training_perc)
            mean_val_loss = metrics.calculate_mean_loss(self.net, validation_loader, max_examples, self.loss_func)
            mean_val_acc = metrics.calculate_accuracy(self.net, validation_loader, max_examples)



        # Set again net to training mode
        self.net.train()

        # Output to the user
        # We don't care about epochs starting in 0, but with iterations is weird
        # ie. epoch 0 it 199 instead of epoch 0 it 200
        print(f"[{epoch} / {epoch_iteration}] <-- ")
        print(f"Training loss: {mean_train_loss}")
        print(f"Validation loss: {mean_val_loss}")
        print(f"Training acc: {mean_train_acc}")
        print(f"Validation acc: {mean_val_acc}")
        print("")

        # Sending this metrics to tensorboard
        curr_it = epoch_iteration * train_loader.batch_size + epoch * len(train_loader.dataset) # The current iteration taking in count
                                                                # that we reset iterations at the end
                                                                # of each epoch

        # Send data to tensorboard
        # We use side-by-side training / validation graphics
        self.tensorboardwriter.add_scalars(
            "Loss",
            {
                "Training loss": mean_train_loss,
                "Validation loss": mean_val_loss,
            },
            curr_it
        )
        self.tensorboardwriter.add_scalars(# Have train / val acc in same graph to compare
            "Accuracy",
            {
                "Training acc": mean_train_acc,
                "Validation acc": mean_val_acc,
            },
            curr_it
        )
        self.tensorboardwriter.flush() # Make sure that writer writes to disk

        # TODO -- this return is wrong, return the actual metrics
        return dict()


    def should_log(self, iteration: int) -> bool:
        if iteration % self.iterations == 0:
            return True

        return False

# TODO -- check if I can use same logger for offline and online triplet training
class TripletLoggerOffline(TrainLogger):
    """
    Custom logger for offline triplet training
    """

    def __init__(self, net: nn.Module, iterations, loss_func):
        """
        Initializes the logger

        Parameters:
        ===========
        net: the net we are testing
        iterations: how many iterations we have to wait to log again
        loss_func: the loss func we are using to train <- Should be some triplet-like loss
        """
        self.iterations = iterations
        self.loss_func = loss_func
        self.net = net

    def log_process(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        epoch_iteration: int
    ) -> Dict[str, float]:

        # Seleccionamos la funcion de perdida
        metric =  metrics.calculate_mean_triplet_loss_offline

        # Empezamos calculando las metricas que queremos mostrar

        # Para tener mas eficiencia en inferencia
        with torch.no_grad():

            # Para tener todavia mas eficiencia en inferencia
            self.net.eval()

            # Funcion de perdida en entrenamiento
            mean_train_loss = metric(self.net, train_loader, self.loss_func)

            # Funcion de perdida en validacion
            mean_val_loss = metric(self.net, validation_loader, self.loss_func)


        # Volvemos a poner la red en modo entrenamiento
        self.net.train()

        # Mostramos las metricas obtenidas
        print(f"[{epoch} / {epoch_iteration}]")
        print(f"\tTraining loss: {mean_train_loss}")
        print(f"\tValidation loss: {mean_val_loss}")
        print("")

        # Devolvemos las funciones de perdida
        return {
            "mean train loss": mean_train_loss,
            "mean validation loss": mean_val_loss,
        }


    def should_log(self, iteration: int) -> bool:
        if iteration % self.iterations == 0 and iteration != 0:
            return True

        return False


class TripletLoggerOnline(TrainLogger):
    """
    Custom logger for online triplet training
    """

    def __init__(
        self,
        net: nn.Module,
        iterations: int,
        loss_func: Callable[[torch.Tensor], float],
        train_percentage: float = 1.0,
        validation_percentage: float = 1.0,
        greater_than_zero: bool = False
    ):
        """
        Initializes the logger

        @param net: the net we are testing
        @param iterations: how many iterations we have to wait to log again
        @param loss_func: the loss func we are using to train <- Should be some triplet-like loss
        @param train_percentage: percentage of the training set we want to use. Less than 1 can be
                                 used for faster computations
        @param validation_percentage: percentage of the training set we want to use. Less than 1 can
                                      be used for faster computations
        @param greater_than_zero: choose if we want to use only greater than zero values for
                                  computing the mean loss
        """
        self.iterations = iterations
        self.loss_func = loss_func
        self.net = net
        self.greater_than_zero = greater_than_zero

        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage

    def log_process(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        epoch_iteration: int
    ) -> Dict[str, float]:

        # This log can be slow so we print this beforehand to have a notion on how slow it is
        print(f"[{epoch} / {epoch_iteration}] <-- ")

        # We are interested in mean triplet loss
        metric = metrics.calculate_mean_loss_function_online

        # Calculamos el numero maximo de ejemplos que evaluar
        train_max_examples = int(len(train_loader.dataset) * self.train_percentage)
        validation_max_examples = int(len(validation_loader.dataset) * self.validation_percentage)

        # Empezamos calculando las metricas que queremos mostrar
        # Para tener mas eficiencia en inferencia
        with torch.no_grad():

            # Para tener todavia mas eficiencia en inferencia
            self.net.eval()

            # Funcion de perdida en entrenamiento
            mean_train_loss = metric(self.net, train_loader, self.loss_func, train_max_examples, self.greater_than_zero)

            # Funcion de perdida en validacion
            mean_val_loss = metric(self.net, validation_loader, self.loss_func, validation_max_examples, self.greater_than_zero)

        # Volvemos a poner la red en modo entrenamiento
        self.net.train()

        # Mostramos las metricas obtenidas
        print(f"\tTraining loss: {mean_train_loss}")
        print(f"\tValidation loss: {mean_val_loss}")
        print("")

        wandb.log({
            "training loss": mean_train_loss,
            "validation loss": mean_val_loss
        })

        # Devolvemos las funciones de perdida
        return {
            "mean train loss": mean_train_loss,
            "mean validation loss": mean_val_loss,
        }

    def should_log(self, iteration: int) -> bool:
        if iteration % self.iterations == 0:
            return True

        return False


class IntraClusterLogger(TrainLogger):
    """
    Logger that logs information about intra cluster information
    This information will be:

    1. Max cluster distance over all clusters
    2. Min cluster distance over all clusters
    3. SDev of cluster distances
    4. Mean of cluster distances

    Given a cluster, its distance is defined as the max distance between two points of that cluster

    """

    def __init__(
        self,
        net: nn.Module,
        iterations: int,
        train_percentage: float = 1.0,
        validation_percentage: float = 1.0,
    ):
        """
        Initializes the logger

        @param net: the net we are testing
        @param iterations: how many iterations we have to wait to log again
        @param train_percentage: percentage of the training set we want to use. Less than 1 can be
                                 used for faster computations
        @param validation_percentage: percentage of the training set we want to use. Less than 1 can
                                      be used for faster computations
        """

        self.net = net
        self.iterations = iterations
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage

    def log_process(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        epoch_iteration: int
    ) -> Dict[str, float]:

        # This log can be slow so we print this beforehand to have a notion on how slow it is
        print(f"[{epoch} / {epoch_iteration}] <-- ")

        # Compute the maximun number of examples to use in the metrics
        train_max_examples = int(len(train_loader.dataset) * self.train_percentage)
        validation_max_examples = int(len(validation_loader.dataset) * self.validation_percentage)

        # Compute the metrics faster
        with torch.no_grad():

            # Compute the metrics faster
            self.net.eval()

            # Get the two metrics
            train_metrics = metrics.compute_cluster_sizes_metrics(train_loader, self.net, train_max_examples)
            validation_metrics = metrics.compute_cluster_sizes_metrics(validation_loader, self.net, validation_max_examples)


        # Get the network in training mode again
        self.net.train()

        # Show obtained metrics
        print(f"\tTraining cluster distances: {train_metrics}")
        print(f"\tValidation cluster distances: {validation_metrics}")
        print("")

        wandb.log({
            "Train Min Cluster Distance": train_metrics["min"],
            "Train Max Cluster Distance": train_metrics["max"],
            "Train Mean Cluster Distance": train_metrics["mean"],
            "Train SD Cluster Distance": train_metrics["sd"],

            "Validation Min Cluster Distance": validation_metrics["min"],
            "Validation Max Cluster Distance": validation_metrics["max"],
            "Validation Mean Cluster Distance": validation_metrics["mean"],
            "Validation SD Cluster Distance": validation_metrics["sd"],
        })

        # If we are using more than one metric logger, in `CompoundLogger`, we need to have unique
        # names for the keys
        renamed_metrics = {
            "Train Min Cluster Distance": train_metrics["min"],
            "Train Max Cluster Distance": train_metrics["max"],
            "Train Mean Cluster Distance": train_metrics["mean"],
            "Train SD Cluster Distance": train_metrics["sd"],

            "Validation Min Cluster Distance": validation_metrics["min"],
            "Validation Max Cluster Distance": validation_metrics["max"],
            "Validation Mean Cluster Distance": validation_metrics["mean"],
            "Validation SD Cluster Distance": validation_metrics["sd"],
        }
        return renamed_metrics


    # TODO -- this method is repeated multiple times
    # TODO -- REFACTOR -- Create base class that does this by default and use it
    def should_log(self, iteration: int) -> bool:
        if iteration % self.iterations == 0:
            return True

        return False

class InterClusterLogger(TrainLogger):
    """
    Logger that logs information about inter cluster information
    This information will be:

    1. Max intercluster distance over all clusters
    2. Min intercluster distance over all clusters
    3. SDev of intercluster distances
    4. Mean of intercluster distances

    Given two clusters, their intercluster distance is the minimun distance between two points, one
    from each of the clusters
    """

    def __init__(
        self,
        net: nn.Module,
        iterations: int,
        train_percentage: float = 1.0,
        validation_percentage: float = 1.0,
    ):
        """
        Initializes the logger

        @param net: the net we are testing
        @param iterations: how many iterations we have to wait to log again
        @param train_percentage: percentage of the training set we want to use. Less than 1 can be
                                 used for faster computations
        @param validation_percentage: percentage of the training set we want to use. Less than 1 can
                                      be used for faster computations
        """

        self.net = net
        self.iterations = iterations
        self.train_percentage = train_percentage
        self.validation_percentage = validation_percentage

    def log_process(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        epoch: int,
        epoch_iteration: int
    ) -> Dict[str, float]:

        # This log can be slow so we print this beforehand to have a notion on how slow it is
        print(f"[{epoch} / {epoch_iteration}] <-- ")

        # Compute the maximun number of examples to use in the metrics
        train_max_examples = int(len(train_loader.dataset) * self.train_percentage)
        validation_max_examples = int(len(validation_loader.dataset) * self.validation_percentage)

        # Compute the metrics faster
        with torch.no_grad():

            # Compute the metrics faster
            self.net.eval()

            # Get the two metrics
            train_metrics = metrics.compute_intercluster_metrics(train_loader, self.net, train_max_examples)
            validation_metrics = metrics.compute_intercluster_metrics(validation_loader, self.net, validation_max_examples)

        # Get the network in training mode again
        self.net.train()

        # Show obtained metrics
        print(f"\tTraining intercluster distances: {train_metrics}")
        print(f"\tValidation intercluster distances: {validation_metrics}")
        print("")

        wandb.log({
            "Train Min Intercluster Distance": train_metrics["min"],
            "Train Max intercluster Distance": train_metrics["max"],
            "Train Mean Intercluster Distance": train_metrics["mean"],
            "Train SD Intercluster Distance": train_metrics["sd"],

            "Validation Min Intercluster Distance": validation_metrics["min"],
            "Validation Max Intercluster Distance": validation_metrics["max"],
            "Validation Mean Intercluster Distance": validation_metrics["mean"],
            "Validation SD Intercluster Distance": validation_metrics["sd"],
        })

        # If we are using more than one metric logger, in `CompoundLogger`, we need to have unique
        # names for the keys
        renamed_metrics = {
            "Train Min Intercluster Distance": train_metrics["min"],
            "Train Max intercluster Distance": train_metrics["max"],
            "Train Mean Intercluster Distance": train_metrics["mean"],
            "Train SD Intercluster Distance": train_metrics["sd"],

            "Validation Min Intercluster Distance": validation_metrics["min"],
            "Validation Max Intercluster Distance": validation_metrics["max"],
            "Validation Mean Intercluster Distance": validation_metrics["mean"],
            "Validation SD Intercluster Distance": validation_metrics["sd"],
        }

        return renamed_metrics

    # TODO -- this method is repeated multiple times
    # TODO -- REFACTOR -- Create base class that does this by default and use it
    def should_log(self, iteration: int) -> bool:

        if iteration % self.iterations == 0:
            return True

        return False
