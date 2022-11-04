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


def main():
    print("==== Running benchmarks for metrics.py ====")
    benchmark_compute_intercluster_metrics()

def __generate_dataloader() -> torch.utils.data.DataLoader:
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

    # Apply data augmentation for having at least 4 images per class
    augmented_dataset = data_augmentation.LazyAugmentatedDataset(
        base_dataset = dataset,
        min_number_of_images = 4,

        # Remember that the trasformation has to be random type
        # Otherwise, we could end with a lot of repeated images
        transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(250, 250)),
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomAutocontrast(),
        ])

    )

    # Now put a loader in front of the augmented dataset
    dataloader = torch.utils.data.DataLoader(
        augmented_dataset,
        batch_size = 3 * 4,
        num_workers = 1,
        pin_memory = True,
        sampler = sampler.CustomSampler(3, 4, augmented_dataset)
    )

    return dataloader

def benchmark_compute_intercluster_metrics():

    print("⌛ compute_intercluster_metrics")

    net = nn.Identity()
    data_loader = __generate_dataloader()
    max_examples = 1_000

    bench_function = lambda: metrics.compute_intercluster_metrics(
        data_loader = data_loader,
        net = net,
        max_examples = max_examples
    )

    benchmark_runner = base.BenchmarkRunner(bench_function, number_experiments = 5, number_runs_per_experiment = 1)
    results = benchmark_runner.run()

    print(f"Results are:\n{results}")
    print("")

if __name__ == "__main__":
    main()
