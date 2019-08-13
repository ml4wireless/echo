from typing import Union, List, Optional

import numpy as np
import torch
from torch import nn

from utils.util_data import get_all_unique_symbols


class Neural(nn.Module):
    def __init__(self, *, 
                 bits_per_symbol:Union[float,int],
                 hidden_layers:List[int] = [50],
                 restrict_energy:Union[float,int] = 1, #0,1,2
                 activation_fn_hidden:str = 'tanh',
                 activation_fn_output:Optional[str] = None,
                 max_amplitude:float = 0.0,
                 **kwargs
                 ):
        super(Neural, self).__init__()   
        self.name = 'neural'
        self.model_type = 'modulator'

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
        

        #MU TRAINING
        assert len(hidden_layers) > 0, "must specify at least one hidden layer"
        layer_dims = [bits_per_symbol]+hidden_layers+[2] # bps --> [hidden layers] --> cartesian 2D
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

        self.restrict_energy = restrict_energy
        self.all_unique_symbols = torch.tensor(get_all_unique_symbols(bits_per_symbol=bits_per_symbol)).float()
        self.bits_per_symbol = bits_per_symbol
        self.max_amplitude = max_amplitude

    def normalize_constellation(self, means):
        #Get average power
        #WARNING: this can cause memory and speed issues for higher modulation orders like QAM 64000
        avg_power = torch.mean(torch.sum((self.base(self.all_unique_symbols))**2,dim=-1))
        #Get normalization factor based on maximum constraint of 1 on average power
        if self.max_amplitude > 0:
            normalization_factor = torch.sqrt((torch.relu(avg_power-self.max_amplitude)+self.max_amplitude) / self.max_amplitude)
        else:
            normalization_factor = torch.sqrt(torch.relu(avg_power-1.0)+1.0)
        #Divide by normalization factor to get modulated symbols
        means = means/normalization_factor

        return means


    def normalize_symbols(self, means):
        avg_power = torch.sqrt(torch.mean(torch.sum(means**2,dim=1)))
        normalization = torch.nn.functional.relu(avg_power-1)+1.0
        means = means / normalization
        return means

    def center_means(self, means):
        center = means.mean(dim=0)
        return means - center

    def center_and_normalize_constellation(self, means):
        const_means = self.center_means(self.base(self.all_unique_symbols))
        avg_power = torch.mean(torch.sum(const_means ** 2,dim=-1))
        #Get normalization factor based on maximum constraint of 1 on average power
        if self.max_amplitude > 0:
            normalization_factor = torch.sqrt((torch.relu(avg_power-self.max_amplitude)+self.max_amplitude) / self.max_amplitude)
        else:
            normalization_factor = torch.sqrt(torch.relu(avg_power-1.0)+1.0)
        #Divide by normalization factor to get modulated symbols
        means = self.center_means(means)/normalization_factor
        return means

    def forward(self, input:torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 2 #input shape should be [N_symbols, bits_per_symbol]
        assert input.shape[1] == self.bits_per_symbol 
        means = self.base(input)
        ###################
        # Normalize outputs
        ###################
        if (self.restrict_energy == 1):
            means = self.normalize_constellation(means)
        elif (self.restrict_energy == 2):
            means = self.normalize_symbols(means)
        elif (self.restrict_energy == 3):
            means = self.center_and_normalize_constellation(means)
        return means

    
    def location_loss(self) -> torch.Tensor:
        return torch.norm(torch.mean(self.base(self.all_unique_symbols))) ** 2

    def l1_loss(self) -> torch.Tensor:
        l1_loss = torch.tensor(0.0)
        for param in self.mu_parameters():
            l1_loss += torch.norm(param, p=1)
        return l1_loss

    def l2_loss(self) -> torch.Tensor:
        l2_loss = torch.tensor(0.0)
        for param in self.mu_parameters():
            l2_loss += torch.norm(param, p=2)
        return l2_loss

    def mu_parameters(self) -> torch.Tensor:
        return self.base.parameters()
