# Script created using code snippets from
# https://github.com/adambielski/siamese-triplet in order to validate that our
# approach is feasible

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import MNIST

from adambielski_lib.trainer import fit

cuda = torch.cuda.is_available()
print(f"=> {cuda=}")

mean, std = 0.1307, 0.3081

train_dataset = MNIST(
    "../data/MNIST",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    ),
)
test_dataset = MNIST(
    "../data/MNIST",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((mean,), (std,))]
    ),
)
n_classes = 10

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


def plot_embeddings(embeddings, targets, xlim=None, ylim=None):
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


from adambielski_lib.datasets import BalancedBatchSampler

# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = BalancedBatchSampler(
    train_dataset.train_labels, n_classes=10, n_samples=25
)
test_batch_sampler = BalancedBatchSampler(
    test_dataset.test_labels, n_classes=10, n_samples=25
)

kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_sampler=train_batch_sampler, **kwargs
)
online_test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_sampler=test_batch_sampler, **kwargs
)

from adambielski_lib.losses import OnlineTripletLoss
from adambielski_lib.metrics import AverageNonzeroTripletsMetric

# Set up the network and training parameters
from adambielski_lib.networks import EmbeddingNet
from adambielski_lib.utils import (  # Strategies for selecting triplets within a minibatch
    AllTripletSelector,
    HardestNegativeTripletSelector,
    RandomNegativeTripletSelector,
    SemihardNegativeTripletSelector,
)

margin = 1.0
embedding_net = EmbeddingNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 20
log_interval = 50

fit(
    online_train_loader,
    online_test_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    metrics=[AverageNonzeroTripletsMetric()],
)

train_embeddings_otl, train_labels_otl = extract_embeddings(train_loader, model)
plot_embeddings(train_embeddings_otl, train_labels_otl)
val_embeddings_otl, val_labels_otl = extract_embeddings(test_loader, model)
plot_embeddings(val_embeddings_otl, val_labels_otl)
