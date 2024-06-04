import os
from copy import deepcopy
from tqdm import tqdm
import argparse

import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from torch.func import jacrev, vmap

import pandas as pd


import random


parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=50000)
parser.add_argument('-dataset', default="cifar")
parser.add_argument('-epochs', default=50)
parser.add_argument('-lr', default=1.0)
parser.add_argument('-measure_every', default=1)
parser.add_argument('-depth', default=8)
parser.add_argument('-model', default='vgg11')
args = parser.parse_args()

for n_, v_ in args.__dict__.items():
    print(f"{n_:<20} : {v_}")

## variables
os.environ['DATA_PATH'] = "/scratch/bbjr/dbeaglehole/Conv"

dataset = args.dataset
n = int(args.n)
NUM_LAYERS = int(args.depth)
epochs = int(args.epochs)
LR = float(args.lr)
MEASURE_EVERY = int(args.measure_every)
model_type = args.model

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

save_path = 'figures'
results_path = 'results'
 
fs=20
fs2=20

#### Generate NFA fig ####



dWs = []
nfas = []
cnfas = []
losses = []

for SEED in [0,1,2]:
    dW_path = os.path.join(results_path, f'{model_type}_dWs_{dataset}_n_{n}_lr_{LR}_seed_{SEED}.pt')
    loss_path = os.path.join(results_path, f'{model_type}_losses_{dataset}_n_{n}_lr_{LR}_epochs_{epochs}_seed_{SEED}.pt')
    nfas_path = os.path.join(results_path, f'{model_type}_nfas_{dataset}_n_{n}_lr_{LR}_seed_{SEED}.pt')
    cnfas_path = os.path.join(results_path, f'{model_type}_cnfas_{dataset}_n_{n}_lr_{LR}_seed_{SEED}.pt')

    nfas.append(torch.load(nfas_path))
    cnfas.append(torch.load(cnfas_path))
    losses.append(torch.load(loss_path))
    dWs.append(torch.load(dW_path))


nfas = torch.stack(nfas)
cnfas = torch.stack(cnfas)
losses = torch.stack(losses)
dWs = torch.stack(dWs)

dW_mean = dWs.mean(dim=0)
nfa_mean = nfas.mean(dim=0)
cnfa_mean = cnfas.mean(dim=0)
losses_mean = losses.mean(dim=0)

dW_stds = dWs.std(dim=0)
nfa_std = nfas.std(dim=0)
cnfa_std = cnfas.std(dim=0)
losses_std = losses.std(dim=0)

loss_fig_title = os.path.join(save_path, f'{model_type}_losses_{dataset}_all_seeds.pdf')
fig, (ax1,ax2) = plt.subplots(1,2)
ax1.semilogy(losses_mean, 'b')
ax1.fill_between(np.arange(len(losses_mean)), losses_mean+losses_std, losses_mean-losses_std, color='b', alpha=0.1)

from matplotlib import colors as mcolors
# colors = ['b','g','r','m','y']
colors = list(mcolors.TABLEAU_COLORS)
for layer in range(NUM_LAYERS):   
    dW = dW_mean[:,layer]
    dW_std = dW_stds[:,layer]
    
    ax2.plot(np.arange(0, len(dW)*MEASURE_EVERY, MEASURE_EVERY), 
             dW, colors[layer], label=f'Layer {layer+1}')
    ax2.fill_between(np.arange(len(dW)), dW+dW_std, dW-dW_std, color=colors[layer], alpha=0.1)

ax2.legend(fontsize=fs2, ncols=1, loc='lower center', bbox_to_anchor=(0.3,-1.0))

ax1.grid()
ax2.grid()

ax1.set_ylabel('Loss')
ax2.set_ylabel('Value')

fig.set_size_inches(12, 6)
fig.savefig(loss_fig_title, format='pdf', bbox_inches='tight')


nfa_fig_title = os.path.join(save_path, f'{model_type}_nfas_{dataset}_all_seeds.pdf')
fig, axes = plt.subplots(1,NUM_LAYERS)
layers_to_plot = range(NUM_LAYERS)
# colors = ['b','g','r','m','y']
for ax, layer in zip(axes, layers_to_plot):
    ax.set_title(f'Layer {layer+1}', fontsize=fs)
    
    nfa = nfa_mean[:,layer]
    cnfa = cnfa_mean[:,layer]
    std = nfa_std[:,layer]
    c_std = cnfa_std[:,layer]
    
    ax.plot(np.arange(0, len(nfa)*MEASURE_EVERY, MEASURE_EVERY), 
             nfa, 'b', label='UC-NFA')
    ax.plot(np.arange(0, len(cnfa)*MEASURE_EVERY, MEASURE_EVERY), 
                 cnfa, 'g', label='C-NFA')

    ax.fill_between(np.arange(0, len(nfa)*MEASURE_EVERY, MEASURE_EVERY), 
                     nfa+std, nfa-std, color='b', alpha=0.1)
    ax.fill_between(np.arange(0, len(cnfa)*MEASURE_EVERY, MEASURE_EVERY), 
                     cnfa+c_std, cnfa-c_std, color='g', alpha=0.1)
    
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    
    ax.set_yticks(torch.linspace(0,1,11))
    
    if layer in [0,4]:
        ax.set_yticklabels([str(x.item())[:3] for x in torch.linspace(0,1,11)])
    else:
        ax.set_yticklabels(['' for x in torch.linspace(0,1,11)])
        
    


axes[0].set_ylabel('Correlation', fontsize=fs)
axes[4].set_ylabel('Correlation', fontsize=fs)

step = epochs // 5
xticks = np.arange(0, epochs + step, step)
for ax in axes:
    ax.set_xticks(xticks)

axes[1].legend(fontsize=fs2, ncols=1, loc='lower center', bbox_to_anchor=(0.3,-0.6))

for ax in axes:
    ax.set_xlabel("Epochs", fontsize=fs)
    ax.grid()

# fig.suptitle("Normalized feature variance throughout training", fontsize=fs)
fig.set_size_inches(64, 6)
fig.savefig(nfa_fig_title, format='pdf', bbox_inches='tight')


