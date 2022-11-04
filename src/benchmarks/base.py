"""
Functions that we're going to use in the benchmarks
There is no standard way of doing benchmarks, so here we define the boilerplate code
"""

from typing import Callable, List, Union
import time

import numpy as np


class BenchmarkResults:
    """
    Simple data class that contains raw results of an experiment and some stats of that raw results
    """

    def __init__(self, raw_results: List[float], mean: float, sd: float):
        self.raw_results: List[float] = raw_results
        self.mean: float = mean
        self.sd: float = sd

    def __str__(self):
        indentation = "    "
        msg = f"BenchmarkResults(\n{indentation}raw_resuls = {self.raw_results},\n{indentation}mean = {self.mean},\n{indentation}sd = {self.sd}\n)"

        return msg

class BenchmarkRunner:
    """
    Runs a benchmark over a given function

    We tried to use `timeit` functionality, but it was unusable
    """

    def __init__(
        self,
        function: Callable,
        number_runs_per_experiment: int = 10_000,
        number_experiments: int = 5
    ):
        """
        @param function: a lambda function that is going to be benchmarked
        @param number_runs_per_experiment: the number of times that the function is called per
               each experiment. The time is accumulated for each experiment
        @param number_experiments: number of experiments that we are going to run
        """
        self.function = function
        self.number_runs_per_experiment = number_runs_per_experiment
        self.number_experiments = number_experiments

    def run(self) -> BenchmarkResults:
        experiment_results: List[float] = []

        for i in range(self.number_experiments):
            print(f"\tRunning experiment {i}")

            start = time.time()

            for j in range(self.number_runs_per_experiment):
                print(f"\t\tRunning repetition {j}")
                self.function()

            end = time.time()

            experiment_results.append(end - start)


        # Now compute some metrics and build the BenchmarkResults object
        mean = np.mean(experiment_results)
        sd = np.std(experiment_results)

        return BenchmarkResults(experiment_results, mean, sd)
