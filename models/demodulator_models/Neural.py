from typing import List, Optional
from torch import nn
import torch
import numpy as np
from typing import List, Optional

class Neural(nn.Module):
    def __init__(self,*,
                 bits_per_symbol:int, 
                 hidden_layers:List[int] = [16],
                 activation_fn_hidden:str = 'tanh',
                 activation_fn_output:Optional[str] = None,
                 **kwargs):

        super(Neural, self).__init__()
        num_classes = 2**bits_per_symbol
        self.name = 'neural'
        self.model_type = 'demodulator'

        activations = {
            'lrelu': nn.LeakyReLU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }          
        assert activation_fn_hidden.lower() in activations.keys(), "demodulator_models/neural.py activation_fn_hidden=%s is unsupported"%activation_fn_hidden
        activation_fn_hidden = activations[activation_fn_hidden.lower()]
        
        if activation_fn_output:
            assert activation_fn_output.lower() in activations.keys(), "demodulator_models/neural.py activation_fn_output=%s is unsupported"%activation_fn_output
            activation_fn_output = activations[activation_fn_output.lower()]
        

        assert len(hidden_layers) > 0, "must specify at least one hidden layer"
        #BASE MODEL'S LAYER DIMENSIONS: cartesian 2D,...hidden layer dims..., num_classes
        num_classes = 2**bits_per_symbol
        layer_dims = [2]+hidden_layers+[num_classes] 
        layers = [] 
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2: #hidden layers
                layers.append(activation_fn_hidden())
            elif activation_fn_output:  #output layer
                layers.append(activation_fn_output())
        self.base = nn.Sequential(*layers) 
        def _init_weights(m):
            if type(m) == nn.Linear:
                y = 1.0/np.sqrt(m.in_features)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0.01)
        self.base.apply(_init_weights) 
        
   
    def forward(self, input:torch.Tensor):
        assert len(input.shape) == 2, "input should be 2D cartesian points with shape [n_samples, 2]"
        #returns logits
        return self.base(input.float())

    def parameters(self):
        return self.base.parameters()

    def l1_loss(self) -> torch.Tensor:
        l1_loss = torch.tensor(0.0)
        for param in self.parameters():
            l1_loss += torch.norm(param, p=1)
        return l1_loss

    def l2_loss(self) -> torch.Tensor:
        l2_loss = torch.tensor(0.0)
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)
        return l2_loss

