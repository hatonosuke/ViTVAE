#! /usr/bin/env python

import torch
import torchvision
from model import ViTVAE
from torch_optimizer import AdaBelief
from collections import defaultdict
from logging import getLogger, basicConfig, INFO
import matplotlib.pyplot as plt


logger = getLogger(__name__)

ROOT='./data'

image_size = 32
patch_size = 2
num_hiddens = 128
dim = 512
depth = 6
heads = 2
mlp_dim = 128
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

@torch.no_grad()
def visualize(model, data, device, nrow=8):
    model.eval()
    xs = prepare(data, device)
    _, variables = model(xs)

    rx = variables['rx'].loc # N, C, H, W

    def _tile(rx):
        N, C, H, W = rx.size()
        rx = ((rx * 255.0/2.0) + 255.0/2.0).to(torch.uint8).cpu()
        num_pad = (nrow - (N % nrow)) % nrow
        if num_pad > 0:
            rx = torch.cat([rx, rx.new_zeros(num_pad, C, H, W)], dim=0)

        N, C, H, W = rx.size()
        rx = rx.view(N//nrow, nrow, C, H, W).permute(0, 3, 1, 4, 2).reshape(N//nrow*H, nrow*W, C)
        return rx.numpy()

    xs = _tile(xs)
    rx = _tile(rx)

    return xs, rx

def train(num_epochs=100, batch_size=32, dataset_size=None):
    basicConfig(level=INFO)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    trainset = torch.from_numpy(torchvision.datasets.CIFAR10(root=ROOT, train=True, download=True).data)
    testset = torch.from_numpy(torchvision.datasets.CIFAR10(root=ROOT, train=False, download=True).data)
    if dataset_size is not None:
        trainset = trainset[:dataset_size]
        testset  = testset[:dataset_size]

    model = ViTVAE(image_size, patch_size, num_hiddens, dim, depth, heads, mlp_dim, dropout=dropout, emb_dropout=emb_dropout).to(device)

    optimizer = AdaBelief(model.parameters(), 1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    for epoch in range(1, num_epochs+1):
        model.train()
        total_losses = defaultdict(float)
        idxes = torch.randperm(len(trainset))

        for i in range(0, len(idxes), batch_size):
            idx = idxes[i:i+batch_size]

            xs = prepare(trainset[idx], device)

            optimizer.zero_grad()
            losses, _ = model(xs)
            losses["loss"].backward()
            optimizer.step()

            losses_append(total_losses, losses, len(idx))

        total_losses = {key: value / len(idxes) for key, value in total_losses.items()}
        logger.info("epoch %s, %s", epoch, ", ".join(["{} = {}".format(key, value) for key, value in total_losses.items()]))
        scheduler.step()

        # validation
        with torch.no_grad():
            model.eval()
            total_losses = defaultdict(float)
            idxes = torch.randperm(len(testset))
            for i in range(0, len(idxes), batch_size):
                idx = idxes[i:i+batch_size]
                xs = prepare(testset[idx], device)

                losses, _ = model(xs)
                losses_append(total_losses, losses, len(idx))

            total_losses = {key: value / len(idxes) for key, value in total_losses.items()}

            logger.info("validation epoch %s, %s", epoch, ", ".join(["{} = {}".format(key, value) for key, value in total_losses.items()]))

    # Visualization
    xs_train, rx_train = visualize(model, trainset[0:8*8], device)
    xs_test, rx_test  = visualize(model, testset[0:8*8], device)

    plt.imsave("rx_train.png", rx_train)
    plt.imsave("xs_train.png", xs_train)
    plt.imsave("rx_test.png", rx_test)
    plt.imsave("xs_test.png", xs_test)

if __name__ == '__main__':
    train(10, 2, 10)
