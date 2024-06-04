import torch
from torch.func import jacrev, vmap
import torch.nn as nn
import models

def mat_cov(A, B):
    A_ = A.reshape(-1).clone()
    B_ = B.reshape(-1).clone().to(A_.device)
    
    A_ -= A_.mean()
    B_ -= B_.mean()
    
    norm1 = A_.norm()
    norm2 = B_.norm()
    
    return (torch.dot(A_, B_) / norm1 / norm2).item()

def get_Ks(net, X, num_layers, width): 
    mb_size = 1024
    Ks = 0.
    with torch.no_grad():
        
        for Xmb in X.split(mb_size, dim=0):
        
            def get_grad_single(x):
                perturbs = []
                for _ in range(num_layers):
                    perturbs.append(torch.zeros((1,width)).cuda())

                def perturb_model(p):
                    return net(x.unsqueeze(0), p).squeeze(0)
                return jacrev(perturb_model)(perturbs)

            grads = vmap(get_grad_single)(Xmb) # list of len L of (n, c, 1, k)
            grads = torch.stack(grads).squeeze(3) # (L, n, c, k)

            grads = grads.moveaxis(-1,1) # (L, k, n, c)
            grads = grads.reshape(num_layers, width, -1)
            Ks += grads @ grads.transpose(1,2)
        
    return Ks


def measure_agop(net, X):
    
    
    n, d = X.shape
    grad = get_jacobian(X)
    grad = grad.reshape(-1, d)
    agop = grad.T @ grad

    return agop

def getW(model, layer):
    model_layer = model.layers[layer]
    return model_layer.weight.data.detach().clone()

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

