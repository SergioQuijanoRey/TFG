import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

import sys
sys.path.append("src/lib")
sys.path.append("src/benchmarks")

import base
import metrics
import sampler
import data_augmentation
import metrics
import sampler
import models
import core


def main():
    print("==== Running benchmarks for metrics.py ====")
    benchmark_compute_intercluster_metrics()

def __generate_dataloader(P, K) -> torch.utils.data.DataLoader:
    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize(
             (0.5, 0.5, 0.5),
             (0.5, 0.5, 0.5)
         ),
    ])
    dataset = torchvision.datasets.LFWPeople(
        root = "./data",
        split = "train",
        download = True,
        transform = transform,
    )

    # Make the dataset smaller so the code doesn't take too much to run
    new_dataset_len = 500
    old_targets = dataset.targets                       # Subset doesnt preserve targets
    dataset = torch.utils.data.Subset(dataset, range(0, new_dataset_len))
    dataset.targets = old_targets[0:new_dataset_len]    # Recover targets

    # Now put a loader in front of the augmented dataset
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = P * K,
        num_workers = 1,
        pin_memory = True,
        sampler = sampler.CustomSampler(P, K, dataset)
    )

    return dataloader

def benchmark_compute_intercluster_metrics():

    print("⌛ compute_intercluster_metrics")

    device = core.get_device()
    net = models.RandomNet(embedding_dimension = 4).to(device)
    data_loader = __generate_dataloader(P = 4, K = 4)
    max_examples = 1_000
    fast_implementation = True

    bench_function = lambda: metrics.compute_intercluster_metrics(
        data_loader = data_loader,
        net = net,
        max_examples = max_examples,
        fast_implementation = fast_implementation,
    )

    benchmark_runner = base.BenchmarkRunner(bench_function, number_experiments = 5, number_runs_per_experiment = 3)
    results = benchmark_runner.run()

    print(f"Results are:\n{results}")
    print("")


if __name__ == "__main__":
    main()
