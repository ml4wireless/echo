import numpy as np 
import torch
from torch import nn
from utils.util_data import integers_to_symbols, symbols_to_integers, get_all_unique_symbols
from typing import Union, Optional

class Demodulator():
    def __init__(self, *,
                 model,
                 bits_per_symbol,
                 optimizer:Optional[str] = 'adam',
                 stepsize_cross_entropy:float = 1e-3, #
                 cross_entropy_weight:float=1.0, #
                 epochs:int = 1,
                 lambda_l1:float = 0.0,
                 lambda_l2:float = 0.0,
                 **kwargs
                ):
        self.epochs = epochs
        self.bits_per_symbol = bits_per_symbol
        self.model = model(bits_per_symbol = bits_per_symbol,
                        **kwargs)
        self.name = self.model.name
        self.lambda_l1=torch.tensor(lambda_l1).float()
        self.lambda_l2=torch.tensor(lambda_l2).float()
        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }
        
        if optimizer and hasattr(self.model, 'parameters') and not hasattr(self.model, "update"):
            assert optimizer.lower() in optimizers.keys(), "demodulator optimizer=%s not supported"%optimizer
            optimizer = optimizers[optimizer.lower()]
            print("Demodulator %s initialized with %s optimizer."%(self.model.name, optimizer.__name__))
            self.cross_entropy_weight = torch.tensor(cross_entropy_weight).float()
            self.param_dicts = [\
                {'params': self.model.parameters(), 'lr': stepsize_cross_entropy},
            ]
            self.optimizer = optimizer(self.param_dicts, lr=stepsize_cross_entropy)
        else:
            print("Demodulator %s initialized WITHOUT an optimizer"%(self.model.name))
            self.optimizer = None
            if hasattr(self.model, 'parameters'):
                self.param_dicts = [\
                    {'params': self.model.parameters(), 'lr': stepsize_cross_entropy},
                ]
            else:
                self.param_dicts = []

    #input cartesian/complex np.ndarray
    #output bit symbols np.ndarray
    def demodulate(self, signal:np.ndarray) -> np.ndarray:
        if len(signal.shape)==2:
            cartesian_points = torch.from_numpy(signal).float()
        elif len(signal.shape)==1:
            cartesian_points = torch.from_numpy(np.stack((signal.real, signal.imag), axis=-1)).float()
        logits = self.model.forward(cartesian_points)
        _, actions = torch.max(logits, 1) #actions are class indexes
        symbols = integers_to_symbols(actions.detach().numpy().astype(int), self.bits_per_symbol)
        return symbols
        
    #input cartesian torch.Tensor
    #output logits for symbols torch.Tensor, map of class id (int) to the bit symbol np.ndarray
    def demodulate_tensor(self, signal:torch.Tensor) -> torch.Tensor:
        cartesian_points= signal
        logits = self.model.forward(cartesian_points)
        class_to_symbol_map = {idx : symbol for idx,symbol in enumerate(get_all_unique_symbols(bits_per_symbol=self.bits_per_symbol))}
        return logits, class_to_symbol_map


    #input cartesian/complex np.ndarray, bit symbols np.ndarray
    def update(self, signal:np.ndarray, true_symbols:np.ndarray, **kwargs):
        if hasattr(self.model, "update"):
            kwargs['signal'] = signal
            kwargs['true_symbols'] = true_symbols
            self.model.update(**kwargs)
            return 
        else:
            assert self.optimizer, "Demodulator is not initialized with an optimizer"
            #train 
            if len(signal.shape)==2:
                cartesian_points = torch.from_numpy(signal).astype(np.float32)
            elif len(signal.shape)==1:
                cartesian_points = torch.from_numpy(np.stack((signal.real.astype(np.float32), signal.imag.astype(np.float32)), axis=-1))
            for _ in range(self.epochs):              
                logits = self.model.forward(cartesian_points)
                target = torch.from_numpy(symbols_to_integers(true_symbols))
                criterion = nn.CrossEntropyLoss()
                loss = self.cross_entropy_weight * torch.mean(criterion(logits, target))
                loss += self.lambda_l1 * self.model.l1_loss()
                loss += self.lambda_l2 * self.model.l2_loss()
                #Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            return

    #input cartesian/complex np.ndarray
    #output class (int) np.ndarray
    def get_demod_grid(self, grid_2d:np.ndarray):
        symbols = self.demodulate(grid_2d)
        classes = symbols_to_integers(symbols)
        return classes

    def get_param_dicts(self):
        return self.param_dicts

    def get_regularization_loss(self):
        if not hasattr(self.model, "update"):
            r_loss =  self.lambda_l1 * self.model.l1_loss()
            r_loss +=  self.lambda_l2 * self.model.l2_loss()
        else:
            r_loss = torch.tensor(0.0).float()
        return r_loss


