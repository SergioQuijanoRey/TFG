import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import sys

from src.lib import loss_functions
sys.path.append("src/lib")
sys.path.append("src/benchmarks")

import base

def main():
    print("==== Running benchmarks for loss_functions.py ====")
    benchmark_precompute_pairwise_distances()

def benchmark_precompute_pairwise_distances():

    print("⌛ precompute_pairwise_distances")

    # Generate a fake dataset
    num_vectors = 1_000
    size_of_vectors = 5
    embeddings = torch.rand(num_vectors, size_of_vectors)

    # Generate the function that we want to benchmark
    triplet_loss = loss_functions.BatchBaseTripletLoss()
    bench_function = lambda: triplet_loss.precompute_pairwise_distances(embeddings, loss_functions.distance_function)

    # Run the benchmarks
    benchmark_runner = base.BenchmarkRunner(
        function = bench_function,
        number_experiments = 5,
        number_runs_per_experiment = 5,
    )
    results = benchmark_runner.run()

    print(f"Results are:\n{results}")
    print("")

if __name__ == "__main__":
    main()
