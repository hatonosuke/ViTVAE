#! /usr/bin/env python

import torch
import torchvision
from model import ViTVAE
from torch_optimizer import AdaBelief
from collections import defaultdict

from logging import getLogger, basicConfig, INFO

logger = getLogger(__name__)

ROOT='./data'

image_size = 32
patch_size = 2
num_hiddens = 128
dim = 512
depth = 6
heads = 8
mlp_dim = 512
channels = 3
dropout = 0.1
emb_dropout = 0.1


def prepare(xs, device):
    if device.type == 'cuda':
        xs = xs.pin_memory().to(device, non_blocking=True)
    return (xs.permute(0, 3, 1, 2).float() - 255./2.)/ (255./2.)


@torch.no_grad()
def losses_append(total_losses, losses, length):
    for key, value in losses.items():
        total_losses[key] += value * length

def train(num_epochs=100, batch_size=32, dataset_size=None):
    basicConfig(level=INFO)

    device = torch.device("cpu")

    trainset = torch.from_numpy(torchvision.datasets.CIFAR10(root=ROOT, train=True, download=True).data)
    testset = torch.from_numpy(torchvision.datasets.CIFAR10(root=ROOT, train=False, download=True).data)
    if dataset_size is not None:
        trainset = trainset[:dataset_size]
        testset  = testset[:dataset_size]

    model = ViTVAE(image_size, patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout=dropout, emb_dropout=emb_dropout).to(device)

    optimizer = AdaBelief(model.parameters(), 1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(1, num_epochs+1):
        total_losses = defaultdict(float)
        idxes = torch.randperm(len(trainset))

        for i in range(len(idxes)):
            idx = idxes[i:i+batch_size]

            xs = prepare(trainset[idx], device)

            optimizer.zero_grad()
            losses, _ = model(xs)
            losses["loss"].backward()
            optimizer.step()

            losses_append(total_losses, losses, len(idx))

        total_losses = {key: value / len(idxes) for key, value in total_losses.items()}

        logger.info("epoch %s, %s", epoch, ", ".join(["{} = {}".format(key, value) for key, value in total_losses.items()]))

        # validation
        with torch.no_grad():
            total_losses = defaultdict(float)
            idxes = torch.randperm(len(testset))
            for i in range(len(idxes)):
                idx = idxes[i:i+batch_size]
                xs = prepare(testset[idx], device)

                losses, _ = model(xs)
                losses_append(total_losses, losses, len(idx))

            total_losses = {key: value / len(idxes) for key, value in total_losses.items()}

            logger.info("validation epoch %s, %s", epoch, ", ".join(["{} = {}".format(key, value) for key, value in total_losses.items()]))


if __name__ == '__main__':
    train(10, 2, 10)
