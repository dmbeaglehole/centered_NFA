import torch
import torch.nn as nn
import torchvision

class Activation(nn.Module):
    def __init__(self):
        super(Activation, self).__init__()
    def forward(self, x):
        return nn.ReLU()(x)
        
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, perturbs=None): 
        use_perturbs = (perturbs is not None)
        ell = 0 
        for layer in self.layers:
            
            # apply perturbs to preactivations
            if use_perturbs:
                if isinstance(layer, Activation):
                    x = x + perturbs[ell]
                    ell += 1
                
            x = layer(x)
        return x

def get_MLP(input_dim, num_classes, width, init, NUM_LAYERS):
    
    inits = [init for _ in range(NUM_LAYERS)]
    layers = [
                nn.Linear(input_dim, width, bias=False), 
                Activation()

             ]
    # nn.init.xavier_uniform_(layers[-2].weight, gain=inits[0])

    for i in range(NUM_LAYERS - 1):
        layers += [
                    nn.Linear(width, width, bias=False),
                    Activation()
                ]
        # nn.init.xavier_uniform_(layers[-2].weight, gain=inits[1+i])

    layers += [nn.Linear(width, num_classes, bias=False)]
    # nn.init.xavier_uniform_(layers[-1].weight, gain=width**-0.5)

    model = MLP(layers)
        
    return model
    
    