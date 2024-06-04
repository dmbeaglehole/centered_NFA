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
import random

import models
import dataset
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-n_train', type=int, default=10000)
parser.add_argument('-n_test', type=int, default=10000)
parser.add_argument('-dataset', default="svhn")
parser.add_argument('-epochs', default=150)
parser.add_argument('-depth', default=5)
parser.add_argument('-lr', default=2.0)
parser.add_argument('-init', default=1.0)
parser.add_argument('-measure_every', default=25)
parser.add_argument('-width', type=int, default=256)
parser.add_argument('-seed', type=int, default=0)
parser.add_argument('-weight_decay', type=float, default=0.)
args = parser.parse_args()

for n_, v_ in args.__dict__.items():
    print(f"{n_:<20} : {v_}")

## variables
os.environ['DATA_PATH'] = "/scratch/bbjr/dbeaglehole/"

dataset_str = args.dataset
n_train = args.n_train
n_test = args.n_test
NUM_LAYERS = int(args.depth)
epochs = int(args.epochs)
LR = float(args.lr)
init = float(args.init)
MEASURE_EVERY = int(args.measure_every)
width = args.width

SEED = int(args.seed)

np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

criterion = nn.MSELoss()

if dataset_str=='cifar':
    train_X, test_X, train_y, test_y = dataset.get_cifar(n_train, n_test, preprocess=True)
elif dataset_str=='cifar100':
    train_X, test_X, train_y, test_y = dataset.get_cifar100(n_train, n_test, preprocess=True)
elif dataset_str=='svhn':
    train_X, test_X, train_y, test_y = dataset.get_svhn(n_train, n_test, preprocess=True)
elif dataset_str=='mnist':
    train_X, test_X, train_y, test_y = dataset.get_mnist(n_train, n_test, preprocess=True)
elif dataset_str=='stl10':
    train_X, test_X, train_y, test_y = dataset.get_stl10(n_train, n_test, preprocess=True)
elif dataset_str=='gtsrb':
    train_X, test_X, train_y, test_y = dataset.get_gtsrb(n_train, n_test, preprocess=True)
    
train_X = train_X.reshape(len(train_X),-1).cuda()
test_X = test_X.reshape(len(test_X),-1).cuda()
train_y = train_y.cuda()
test_y = test_y.cuda()

n, input_dim = train_X.shape
_, num_classes = train_y.shape

print("n_train:", n,"n_test:",len(test_X))
    
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
            return len(self.y)

    def __getitem__(self, idx):
            X_mb = self.X[idx]
            y_mb = self.y[idx]
            return (X_mb, y_mb)
    
mb_size = 128
train_loader = torch.utils.data.DataLoader(MyDataset(train_X, train_y), batch_size=mb_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(MyDataset(test_X, test_y), batch_size=mb_size, shuffle=True)

  
def train_step(net, optimizer, train_loader):
    net.train()
    train_loss = 0.
    num_batches = len(train_loader)

    for batch_idx, batch in enumerate(train_loader):
        
        optimizer.zero_grad()
        inputs, labels = batch
        targets = labels
        output = net(Variable(inputs)).float()
        target = Variable(targets).float()
        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        train_loss += loss.cpu().data.numpy() * len(inputs)
    train_loss = train_loss / len(train_loader.dataset)
    return train_loss


def val_step(net, val_loader):
    net.eval()
    val_loss = 0.
    for batch_idx, batch in enumerate(val_loader):
        inputs, labels = batch
        targets = labels
        with torch.no_grad():
            output = net(Variable(inputs))
            target = Variable(targets)
        loss = criterion(output, target)
        val_loss += loss.cpu().data.numpy() * len(inputs)
    val_loss = val_loss / len(val_loader.dataset)
    return val_loss


PRINT_EVERY = MEASURE_EVERY


def train_network(net, train_loader, test_loader, lr=LR, num_epochs=epochs):
    
    W0s = []
    for layer in range(NUM_LAYERS):
        W0s.append(utils.getW(net, layer*2))
        
        
    params = 0
    for i, param in enumerate(list(net.parameters())):
        size = 1
        for j in range(len(param.size())):
            size *= param.size()[j]
            params += size

    print("NUMBER OF PARAMS: ", params)

    losses = []
    nfas = []
    cnfas = []
    dWs = []
    
    best_loss = float("inf")
    for i in range(num_epochs+1):
        
        if i==0:
            LR = 1e-4
        else:
            LR = lr 
            
            
        LR = LR * num_classes / 10
            
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, weight_decay=args.weight_decay)
        
        train_loss = train_step(net, optimizer, train_loader)
        test_loss = val_step(net, test_loader)
     
        losses.append(train_loss)

        if test_loss < best_loss:
            best_loss = test_loss
        
        
        if i%PRINT_EVERY==0:
            print("Epoch: ", i,
                  "Train Loss: ", train_loss, "Test Loss: ", test_loss,
                  "Best Test Loss: ", best_loss)
            
        if i%MEASURE_EVERY==0:
            nfa = torch.zeros(NUM_LAYERS).to(train_X.device)
            cnfa = torch.zeros(NUM_LAYERS).to(train_X.device)
            dW = torch.zeros(NUM_LAYERS).to(train_X.device)
            
            
            Ks = utils.get_Ks(net, train_X, NUM_LAYERS, width)
            for layer in range(NUM_LAYERS):
                
                K = Ks[layer]
                W = utils.getW(net, layer*2)
                W0 = W0s[layer]
                
                Wb = W - W0
                dW[layer] = Wb.norm()/W.norm()

                nfm = W.T@W
                c_nfm = Wb.T @ Wb

                agop = W.T @ K @ W
                c_agop = Wb.T @ K @ Wb

                nfa[layer] = utils.mat_cov(agop, nfm)
                cnfa[layer] = utils.mat_cov(c_agop, c_nfm)
            
            print("nfa",nfa)
            print("cnfa",cnfa)
            print("||dW||/||W||",dW)
            print()
            nfas.append(nfa)
            cnfas.append(cnfa)
            dWs.append(dW)
            
    nfas = torch.stack(nfas)
    cnfas = torch.stack(cnfas)
    dWs = torch.stack(dWs)
    
    torch.save(nfas.cpu(), f'results/mlp_nfas_{dataset_str}_n_{n_train}_lr_{lr}_epochs_{epochs}_width_{width}_seed_{SEED}.pt')
    torch.save(cnfas.cpu(), f'results/mlp_cnfas_{dataset_str}_n_{n_train}_lr_{lr}_epochs_{epochs}_width_{width}_seed_{SEED}.pt')
    torch.save(dWs.cpu(), f'results/mlp_dWs_{dataset_str}_n_{n_train}_lr_{lr}_epochs_{epochs}_width_{width}_seed_{SEED}.pt')
    
    return net, losses, nfas, cnfas

model = models.get_MLP(input_dim, num_classes, width, init, NUM_LAYERS)

model.cuda()

model, losses, nfas, cnfas = train_network(model, train_loader, test_loader)

torch.save(torch.tensor(losses), f'results/mlp_losses_{dataset_str}_n_{n_train}_lr_{LR}_epochs_{epochs}_width_{width}_seed_{SEED}.pt')
