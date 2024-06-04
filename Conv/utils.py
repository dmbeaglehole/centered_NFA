import torch
from torch.func import jacrev, vmap
import torch.nn as nn

import models
from models import MyConv

def mat_cov(A, B):
    A_ = A.reshape(-1).clone()
    B_ = B.reshape(-1).clone().to(A_.device)
    
    A_ -= A_.mean()
    B_ -= B_.mean()
    
    norm1 = A_.norm()
    norm2 = B_.norm()
    
    return (torch.dot(A_, B_) / norm1 / norm2).item()

def get_perturbs(model, sizes, device='cuda'):
    perturbs = []
    ell = 0
    for layer in model.layers:
        if isinstance(layer, MyConv):
            kl, Pl, Ql = sizes[ell]
            p = torch.zeros((1,kl,Pl,Ql)).to(device)
            perturbs.append(p)
            ell += 1
    return perturbs

def get_num_conv_layers(model):
    num_conv_layers = 0
    for layer in model.layers:
        if isinstance(layer, MyConv):
            num_conv_layers += 1
    return num_conv_layers

def get_sizes(model):
    sizes = []
    ell = 0
    P = 32
    Q = 32
    for layer in model.layers:
        if isinstance(layer, MyConv):
            kl = layer.weight.data.shape[0]
            sizes.append((kl, P, Q))
            ell += 1
        elif isinstance(layer, nn.MaxPool2d):
            P = P//2
            Q = Q//2
    return sizes

def get_Ks(net, X):
    with torch.no_grad():
        model = models.perturbify_model(net)
        num_conv_layers = get_num_conv_layers(model)
        sizes = get_sizes(model)

        def get_jacobian(X_):
            def perturb_grad_single(x_):
                perturbs = get_perturbs(model, sizes)

                def perturb_model(p):
                    return model(x_.unsqueeze(0), p).squeeze(0)

                return jacrev(perturb_model)(perturbs)

            return vmap(perturb_grad_single)(X_)

        grad = get_jacobian(X)
        Ks = []
        for g in grad:
            g_ = g.moveaxis(3,0)
            g_ = g_.reshape(len(g_),-1)
            Ks.append(g_@g_.T)
    return Ks

def getConvW(net, layer_idx):
    model = models.perturbify_model(net)
    ell = 0
    for layer in model.layers:
        if isinstance(layer, MyConv):
            if ell==layer_idx:
                return layer.weight.data.detach().clone()
            ell += 1  
    return

def get_conv_layers(net):
    model = models.perturbify_model(net)
    conv_layers = []
    for layer in model.layers:
        if isinstance(layer, MyConv):
            conv_layers.append(layer.weight.data.detach().clone())
    return conv_layers

def get_fmap(net_, last_layer):
    if last_layer==0:
        return nn.Identity()
    return nn.Sequential(*net_.layers[:last_layer]).eval()
    
def get_submodel(net_, last_layer):
    return nn.Sequential(*net_.layers[last_layer:]).eval()

def get_fmap_resnet(net_, last_layer):
    resnet = net_[0]
    mlp = net_[1]
    return nn.Sequential(*[resnet, get_fmap(mlp, last_layer)]).eval()

def get_fmap_vgg(net_, last_layer):
    vgg = net_[0]
    mlp = net_[1]
    return nn.Sequential(*[vgg, get_fmap(mlp, last_layer)]).eval()

def getW(model, layer):
    model_layer = model.layers[layer]
    return model_layer.weight.data
