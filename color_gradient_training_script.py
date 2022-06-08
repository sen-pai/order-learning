import os
import argparse
import numpy as np

import matplotlib.pyplot as plt

plt.style.use("seaborn-white")
plt.rc("font", size=8)

# from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from itertools import chain
from typing import List

from models import CNNbase, Comparator
from datasets import ColorGradientMain, ColorGradientComp


# collect commandline arguments
parser = argparse.ArgumentParser(description="simple color gradient comparator")
parser.add_argument(
    "--exp-name",
    default="simple_exp",
    help="Experiment name",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    help="number of training epochs",
)
args = parser.parse_args()

# create instance of datasets and corresponding dataloaders
dataset_main = ColorGradientMain()
dataset_comp = ColorGradientComp()

dataloader_main = DataLoader(dataset_main, batch_size=10, shuffle=True)
dataloader_comp = DataLoader(dataset_comp, batch_size=10, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create an instance of the models
cnn_base = CNNbase().to(device)
comparator = Comparator().to(device)

# optimize both models together
optimizer = optim.Adam(chain(cnn_base.parameters(), comparator.parameters()), lr=1e-3)

# we will use CEM loss to train the models
criterion = nn.CrossEntropyLoss()

# logging and plotting dir
current_dir = os.getcwd()
exp_dir = os.path.join(current_dir, "experiments", args.exp_name)
plot_dir = os.path.join(exp_dir, "plots")

# create exp dir if it doesnt exist
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)


def plot_index_gen(start, increment, list_len) -> List:
    gen_list = []
    for i in range(list_len):
        if i == 0:
            gen_list.append(start)
        else:
            gen_list.append(gen_list[-1]+increment)
    return gen_list


def plot_predicted(
    x_i: np.array,
    x_j: np.array,
    order: np.array,
    pred_order: np.array,
    save_path: os.PathLike,
):
    batch_size = x_i.shape[0]

    fig = plt.figure()

    x_i_indices = plot_index_gen(1, 4, batch_size)
    for i, ind in enumerate(x_i_indices):
        ax = fig.add_subplot(batch_size, 4, ind)
        ax.imshow(np.moveaxis(np.array(x_i[i]), 0, 2))
        plt.axis("off")

    x_j_indices = plot_index_gen(2, 4, batch_size)
    for i, ind in enumerate(x_j_indices):
        ax = fig.add_subplot(batch_size, 4, ind)
        ax.imshow(np.moveaxis(np.array(x_j[i]), 0, 2))
        plt.axis("off")
    
    order_indices = plot_index_gen(3, 4, batch_size)
    for i, ind in enumerate(order_indices):
        ax = fig.add_subplot(batch_size, 4, ind)
        ax.text(0.5, 0.5, str(order[i]),fontsize=18, ha='center')
        plt.axis("off")
    
    pred_order_indices = plot_index_gen(4, 4, batch_size)
    for i, ind in enumerate(pred_order_indices):
        ax = fig.add_subplot(batch_size, 4, ind)
        ax.text(0.5, 0.5, str(pred_order[i]),fontsize=18, ha='center')
        plt.axis("off")
    

    fig.savefig(save_path, dpi=200)


# training and validation epochs
for epoch in range(args.epochs):

    # training loop
    cnn_base.train()
    comparator.train()

    for x_i, x_j, order in dataloader_comp:

        x_i, x_j, order = x_i.to(device), x_j.to(device), order.to(device)

        x_i_features = cnn_base(x_i.float())
        x_j_features = cnn_base(x_j.float())

        pred_order = comparator(x_i_features, x_j_features)

        optimizer.zero_grad()
        loss = criterion(pred_order, order)

        print(loss)
        loss.backward()
        optimizer.step()

    # eval and save plots
    cnn_base.eval()
    comparator.eval()
    count = 0

    for (x_i, rank_i), (x_j, rank_j) in zip(dataloader_main, dataloader_main):

        # TODO: use numpy to avoid this loop
        order = []
        for r_i, r_j in zip(rank_i, rank_j):
            if r_i == r_j:
                order.append(1)
            elif r_i < r_j:
                order.append(0)
            elif r_i > r_j:
                order.append(2)

        x_i_features = cnn_base(x_i.float())
        x_j_features = cnn_base(x_j.float())

        pred_order = comparator(x_i_features, x_j_features)
        # pred_order = F.log_softmax(pred_order, 1)
        pred_order = torch.argmax(pred_order, 1).numpy()
        plot_predicted(x_i, x_j, order, pred_order, os.path.join(plot_dir, str(count)+".png"))
        count += 1
