import os
from copy import deepcopy
from tqdm import tqdm


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
import dataset
import utils
from vgg import vgg11_bn
import models

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n_train', type=int, default=100)
parser.add_argument('-n_test', type=int, default=100)
parser.add_argument('-dataset', default="cifar")
parser.add_argument('-epochs', default=100)
parser.add_argument('-lr', default=1.0)
parser.add_argument('-momentum', type=float, default=0.9)
parser.add_argument('-opt', default='sgd')
parser.add_argument('-measure_every', default=10)
parser.add_argument('-model', default='vgg11')
parser.add_argument('-seed', type=int, default=0)
args = parser.parse_args()

for n_, v_ in args.__dict__.items():
    print(f"{n_:<20} : {v_}")

## variables
os.environ['DATA_PATH'] = "/scratch/bbjr/dbeaglehole/"

dataset_str = args.dataset
epochs = int(args.epochs)
LR = float(args.lr)
MEASURE_EVERY = int(args.measure_every)
n_train = int(args.n_train)
n_test = int(args.n_test)
SEED = int(args.seed)
momentum = args.momentum

np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

model_type = args.model
criterion = nn.MSELoss()

if dataset_str=='cifar':
    train_X, test_X, train_y, test_y = dataset.get_cifar(n_train, n_test, preprocess=True)
elif dataset_str=='cifar100':
    train_X, test_X, train_y, test_y = dataset.get_cifar100(n_train, n_test, preprocess=True)
elif dataset_str=='svhn':
    train_X, test_X, train_y, test_y = dataset.get_svhn(n_train, n_test, preprocess=True)
elif dataset_str=='emnist':
    train_X, test_X, train_y, test_y = dataset.get_emnist(n_train, n_test, preprocess=True)
elif dataset_str=='stl10':
    train_X, test_X, train_y, test_y = dataset.get_stl10(n_train, n_test, preprocess=True)
elif dataset_str=='gtsrb':
    train_X, test_X, train_y, test_y = dataset.get_gtsrb(n_train, n_test, preprocess=True)
    
train_X = train_X.cuda()
test_X = test_X.cuda()
train_y = train_y.cuda()
test_y = test_y.cuda()

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
    
mb_size = 256
train_loader = torch.utils.data.DataLoader(MyDataset(train_X, train_y), batch_size=mb_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(MyDataset(test_X, test_y), batch_size=mb_size, shuffle=True)

  
def train_step(net, optimizer, train_loader):
    net.train()
    train_loss = 0.
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(train_loader):
        
        optimizer.zero_grad()
        inputs, labels = batch
        
        # inputs = inputs.cuda()
        targets = labels
        output = net(Variable(inputs)).float()
        # inputs = inputs.cpu()
        
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

    params = 0
    for i, param in enumerate(list(net.parameters())):
        size = 1
        for j in range(len(param.size())):
            size *= param.size()[j]
            params += size

    print("NUMBER OF PARAMS: ", params)
    
    
    W0 = utils.get_conv_layers(net)
    num_conv_layers = utils.get_num_conv_layers(models.perturbify_model(net))

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
            
        optimizer = torch.optim.SGD(net.parameters(), lr=LR, momentum=momentum)
        
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
            nfa = torch.zeros(num_conv_layers).to(train_X.device)
            cnfa = torch.zeros(num_conv_layers).to(train_X.device)
            dW = torch.zeros(num_conv_layers).to(train_X.device)
            
            Ks = utils.get_Ks(net, train_X[:1000])
            for layer in range(num_conv_layers):
                K = Ks[layer]
                W = utils.getConvW(model, layer)
                
                Wb = W - W0[layer]

                nfm = W.T@W
                c_nfm = Wb.T @ Wb

                agop = W.T @ K @ W
                c_agop = Wb.T @ K @ Wb

                nfa[layer] = utils.mat_cov(agop, nfm)
                cnfa[layer] = utils.mat_cov(c_agop, c_nfm)
                dW[layer] = Wb.norm() / W.norm()

            nfas.append(nfa)
            cnfas.append(cnfa)
            dWs.append(dW)
            print("NFA:", nfa)
            print("CNFA:", cnfa)
            print('dW:', dW)
            print()
    
    nfas = torch.stack(nfas)
    cnfas = torch.stack(cnfas)
    dWs = torch.stack(dWs)
    
    torch.save(nfas.cpu(), f'results/{model_type}_nfas_{dataset_str}_n_{n_train}_lr_{lr}_epochs_{epochs}_seed_{SEED}.pt')
    torch.save(cnfas.cpu(), f'results/{model_type}_cnfas_{dataset_str}_n_{n_train}_lr_{lr}_epochs_{epochs}_seed_{SEED}.pt')
    torch.save(dWs.cpu(), f'results/{model_type}_dWs_{dataset_str}_n_{n_train}_lr_{lr}_epochs_{epochs}_seed_{SEED}.pt')
    
    return net, losses, nfas, cnfas

if model_type=='vgg11':
    model = vgg11_bn()
    model = models.remove_bn(model)

print(model)
print(models.perturbify_model(model))
model.cuda()

model, losses, nfas, cnfas = train_network(model, train_loader, test_loader)

torch.save(torch.tensor(losses), f'results/{model_type}_losses_{dataset_str}_n_{n_train}_lr_{LR}_epochs_{epochs}_seed_{SEED}.pt')
