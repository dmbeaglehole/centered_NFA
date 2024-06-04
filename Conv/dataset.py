import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm
import random
import os

from math import log, sqrt


def one_hot_data(dataset, num_classes, num_samples):
    Xs = []
    ys = []

    for ix in range(min(len(dataset),num_samples)):
        X,y = dataset[ix]
        Xs.append(X)
        
        ohe_y = torch.zeros(num_classes)
        ohe_y[y] = 1
        ys.append(ohe_y)

    return torch.stack(Xs), torch.stack(ys)

def get_binary(dataset, classes):
    c1, c2 = classes
    
    binary_dataset = []
    for ix in tqdm(range(len(dataset))):
        X,y = dataset[ix]
        
        if y==c1:
            binary_dataset.append((X,0))
        elif y==c2:
            binary_dataset.append((X,1))

    return binary_dataset

def process(train_X, test_X, train_y, test_y, NUM_CLASSES, preprocess):
    n_train, c, P, Q = train_X.shape
    n_test = len(test_X)
    
    if preprocess:
        print("Preprocessing")
        shift = (1. / NUM_CLASSES if NUM_CLASSES > 1 else 0.5)
        train_y -= shift
        test_y -= shift

        # Normalize by precomputed per channel mean/std from training images
        mean = torch.mean(train_X, dim=(0,2,3)).reshape(1,-1,1,1)
        std = torch.std(train_X, dim=(0,2,3)).reshape(1,-1,1,1)

        train_X = (train_X - mean) / std
        test_X = (test_X - mean) / std
    
    return train_X, test_X, train_y, test_y

def get_cifar(n_train, n_test, preprocess=True):

    NUM_CLASSES = 10

    path = os.environ["DATA_PATH"] + "cifar10/"
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root=path,
                                            train=True,
                                            transform=transform,
                                            download=False)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)

    testset = torchvision.datasets.CIFAR10(root=path,
                                           train=False,
                                           transform=transform,
                                           download=False)

    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)

    
    train_X, test_X, train_y, test_y = process(train_X, test_X, train_y, test_y, 
                                               NUM_CLASSES, preprocess)
        
    return train_X, test_X, train_y, test_y

def get_cifar100(n_train, n_test, preprocess=True):

    NUM_CLASSES = 100
    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + "cifar100/"

    trainset = torchvision.datasets.CIFAR100(root=path,
                                            train=True,
                                            transform=transform,
                                            download=True)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)

    testset = torchvision.datasets.CIFAR100(root=path,
                                           train=False,
                                           transform=transform,
                                           download=True)

    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)
    
    
    train_X, test_X, train_y, test_y = process(train_X, test_X, train_y, test_y, 
                                               NUM_CLASSES, preprocess)
    

    return train_X, test_X, train_y, test_y

def get_svhn(n_train, n_test, preprocess=True):

    NUM_CLASSES = 10

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + 'SVHN/'

    trainset = torchvision.datasets.SVHN(root=path,
                                         split='train',
                                         transform=transform,
                                         download=False)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)

    testset = torchvision.datasets.SVHN(root=path,
                                        split='test',
                                        transform=transform,
                                        download=False)

    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)
    
    train_X, test_X, train_y, test_y = process(train_X, test_X, train_y, test_y, 
                                               NUM_CLASSES, preprocess)

    return train_X, test_X, train_y, test_y


def get_stl10(n_train, n_test, preprocess=True):

    NUM_CLASSES = 10

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + 'STL10/'

    trainset = torchvision.datasets.STL10(root=path,
                                         split='train',
                                         transform=transform,
                                         download=True)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)

    testset = torchvision.datasets.STL10(root=path,
                                        split='test',
                                        transform=transform,
                                        download=True)


    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)
    
    train_X, test_X, train_y, test_y = process(train_X, test_X, train_y, test_y, 
                                               NUM_CLASSES, preprocess)

    return train_X, test_X, train_y, test_y


def get_gtsrb(n_train, n_test, preprocess=True):

    NUM_CLASSES = 43

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
    )

    path = os.environ["DATA_PATH"] + 'STL10/'

    trainset = torchvision.datasets.GTSRB(root=path,
                                         split='train',
                                         transform=transform,
                                         download=True)

    train_X, train_y = one_hot_data(trainset, NUM_CLASSES, num_samples=n_train)

    testset = torchvision.datasets.GTSRB(root=path,
                                        split='test',
                                        transform=transform,
                                        download=True)

    test_X, test_y = one_hot_data(testset, NUM_CLASSES, num_samples=n_test)
    
    train_X, test_X, train_y, test_y = process(train_X, test_X, train_y, test_y, 
                                               NUM_CLASSES, preprocess)

    return train_X, test_X, train_y, test_y
