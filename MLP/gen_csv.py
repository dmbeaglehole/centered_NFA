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
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

## variables
os.environ['DATA_PATH'] = "/scratch/bbjr/dbeaglehole/MLP"
save_path = 'figures'
results_path = 'results'
 
fs=20
fs2=20

all_args = [
    ("cifar", 50000, 5, 150, 2.0, 256, 1),
    ("cifar100", 50000, 5, 150, 2.0, 256, 1),
    ("svhn", 50000, 5, 150, 2.0, 256, 1),
    ("stl10", 50000, 5, 150, 0.1, 256, 1),
    ("gtsrb", 50000, 5, 150, 0.1, 256, 1),
    ("mnist", 50000, 5, 50, 10.0, 256, 1)
]

init_nfas = []
init_cnfas = []

final_nfas = []
final_cnfas = []
    
for arg in all_args:
    dataset, n, NUM_LAYERS, epochs, LR, width, MEASURE_EVERY = arg

    for SEED in [0,1,2]:
        nfas_path = os.path.join(results_path, f'mlp_nfas_{dataset}_n_{n}_lr_{LR}_epochs_{epochs}_width_{width}_seed_{SEED}.pt')
        cnfas_path = os.path.join(results_path, f'mlp_cnfas_{dataset}_n_{n}_lr_{LR}_epochs_{epochs}_width_{width}_seed_{SEED}.pt')

        nfa = torch.load(nfas_path)
        cnfa = torch.load(cnfas_path)
        
        init_nfas.append(nfa[0])
        init_cnfas.append(cnfa[0])
        final_nfas.append(nfa[-1])
        final_cnfas.append(cnfa[-1])
        
init_nfas = torch.stack(init_nfas)
init_cnfas = torch.stack(init_cnfas)
final_nfas = torch.stack(final_nfas)
final_cnfas = torch.stack(final_cnfas)

                         
init_nfa_mean = init_nfas.mean(dim=0)
init_cnfa_mean = init_cnfas.mean(dim=0)
# init_nfa_std = init_nfas.std(dim=0)
# init_cnfa_std = init_cnfas.std(dim=0)

final_nfa_mean = final_nfas.mean(dim=0)
final_cnfa_mean = final_cnfas.mean(dim=0)
# final_nfa_std = final_nfas.std(dim=0)
# final_cnfa_std = final_cnfas.std(dim=0)


def save_to_csv(t, fname):
    t_np = t.numpy() #convert to Numpy array
    df = pd.DataFrame(t_np) #convert to a dataframe
    df.to_csv('csv_results/' + fname, index=False) #save to file
    
    
save_to_csv(init_nfas, f'mlp_all_datasets_nfas_init.csv')
save_to_csv(init_cnfas, f'mlp_all_datasets_cnfas_init.csv')
save_to_csv(final_nfas, f'mlp_all_datasets_nfas_final.csv')
save_to_csv(final_cnfas, f'mlp_all_datasets_cnfas_final.csv')

# save_to_csv(init_nfa_mean, f'mlp_all_datasets_nfa_mean_init.csv')
# save_to_csv(init_cnfa_mean, f'mlp_all_datasets_cnfa_mean_init.csv')
# save_to_csv(init_nfa_std, f'mlp_all_datasets_nfa_std_init.csv')
# save_to_csv(init_cnfa_std, f'mlp_all_datasets_cnfa_std_init.csv')

# save_to_csv(final_nfa_mean, f'mlp_all_datasets_nfa_mean_final.csv')
# save_to_csv(final_cnfa_mean, f'mlp_all_datasets_cnfa_mean_final.csv')
# save_to_csv(final_nfa_std, f'mlp_all_datasets_nfa_std_final.csv')
# save_to_csv(final_cnfa_std, f'mlp_all_datasets_cnfa_std_final.csv')




