"""
Functions that we're going to use in the benchmarks
There is no standard way of doing benchmarks, so here we define the boilerplate code
"""

import timeit
from typing import Callable, List, Union

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
        msg =f"BenchmarkResults(\n{indentation}raw_resuls = {self.raw_results},\n{indentation}mean = {self.mean},\n{indentation}sd = {self.sd}\n)"

        return msg


def benchmark_function(
        function: Union[Callable, str],
        repeat: int = 5,
        number: int = 10_000,
        setup: Union[Callable, str] = "pass"
) -> BenchmarkResults:
    """
    Given a function and some repetition parameters, runs a benchmark over that function

    @param function lambda function that we are going to benchmark or a string containing the code
           to execute
    @param repeat the number of experiments that we are going to launch
    @param number the number of times the function is called in each experiment
    @param setup lambda function or string containing setup code (ie. var instantiation)

    @returns a BenchmarkResults object containing the raw results and some stats
    """

    # Create a timer that we are going to use to benchmark the function
    timer = timeit.Timer(function, setup = setup)

    # Run the executions
    results = timer.repeat(repeat = repeat, number = number)

    # Compute some stats over the results
    mean: float = np.mean(results)
    sd: float = np.std(results)

    return BenchmarkResults(results, mean, sd)
