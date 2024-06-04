import torch
import torch.nn as nn
from einops import rearrange

def patchify(x, patch_size):
    n,c,P,Q = x.shape
    # print("x1",x.shape)
    x = nn.functional.pad(x, (1,1,1,1))
    # print("x2",x.shape, "ps", patch_size)
    x = x.unfold(2, patch_size, 1).unfold(3, patch_size, 1) # (n, c, P, Q, w, h)
    # print("x3",x.shape)
    x = rearrange(x, 'n C P Q w h -> n P Q (C w h)')
    return x

class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()
    def forward(self, x):
        return nn.ReLU()(x)

class MyConv(nn.Module):
    def __init__(self, conv):
        super(MyConv, self).__init__()        
        w = conv.weight.data
        self.k = w.shape[0]
        self.ps = 3
        
        w = w.reshape(len(w), -1)
        self.weight = nn.Parameter(w)
        
    def forward(self, x, perturbs):
        n, _, P, Q = x.shape
        # print("x",x.shape)
        x = patchify(x, patch_size=self.ps) # (n, P, Q, c*w*h)
        # print("x2",x.shape)
        x = x@self.weight.T
        # print("x3",x.shape)
        x = rearrange(x, 'n P Q k -> n k P Q')
        x = x + perturbs
        return x
    
class Network(nn.Module):
    def __init__(self, layers):
        super(Network, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, perturbs): 
        ell = 0
        for layer in self.layers:
            # print("x",x.shape)
            if isinstance(layer, MyConv):
                x = layer(x, perturbs[ell])
                ell += 1
            else:
                x = layer(x)
        return x
    
def perturbify_model(model):
    
    model.eval()
    
    l1 = [module for module in model.features.modules() if not isinstance(module, nn.Sequential)]
    l2 = [module for module in model.avgpool.modules() if not isinstance(module, nn.Sequential)]
    l3 = [module for module in model.classifier.modules() if not isinstance(module, nn.Sequential)]
    layers = l1+l2+l3
    
    new_layers = []
    for layer in layers:
        if isinstance(layer, nn.Conv2d):
            new_conv = MyConv(layer)
            new_layers.append(new_conv)
        else:
            new_layers.append(layer)
        
        if isinstance(layer, nn.AdaptiveAvgPool2d):
            new_layers.append(nn.Flatten())
            
    return Network(new_layers)

def remove_bn(model):
    # features, avgpool, classifier
    feats = model.features
    new_feats = []
    for layer in feats.modules():
        if isinstance(layer, nn.Sequential):
            continue
        if not isinstance(layer, nn.BatchNorm2d):
            new_feats.append(layer)
            
    new_feats = nn.Sequential(*new_feats)
    model.features = new_feats
    return model
            
        
    