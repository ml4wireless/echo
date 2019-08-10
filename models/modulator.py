from utils.util_data import cartesian_2d_to_complex, integers_to_symbols
import numpy as np 
import torch
from torch import nn
from typing import Union, Optional

class Modulator():
    def __init__(self,*, 
                model,
                bits_per_symbol:Union[float,int] ,
                lambda_baseline:float = 0.0, #used in update method
                lambda_center:float = 0.0,   #used in update method
                lambda_l1:float = 0.0,       #used in update method
                lambda_l2:float = 0.0,       #used in update method
                std_min:float = 1e-3,
                std_max:float = 1e2,
                initial_std:float = 5e-1,
                optimizer:Optional[str] = 'adam',
                stepsize_mu:float=    0.0,
                stepsize_sigma:float= 0.0,
                **kwargs):

        self.model = model(bits_per_symbol = bits_per_symbol, **kwargs)
        self.name = self.model.name
        self.bits_per_symbol = bits_per_symbol
        self.std_min=torch.tensor(std_min).float()
        self.std_max=torch.tensor(std_max).float()
        self.lambda_l1=torch.tensor(lambda_l1).float()
        self.lambda_l2=torch.tensor(lambda_l2).float()
        self.lambda_center=torch.tensor(lambda_center).float()
        self.lambda_baseline=torch.tensor(lambda_baseline).float()
        self.all_symbols = integers_to_symbols(np.arange(0, 2**bits_per_symbol), bits_per_symbol)
        self.std = nn.Parameter(
            torch.from_numpy(np.array([initial_std, initial_std]).astype(np.float32)),
            requires_grad=True
        )

        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }

        
        if optimizer and hasattr(self.model, "mu_parameters") and not hasattr(self.model, "update"):
            assert optimizer.lower() in optimizers.keys(), "modulator optimizer=%s not supported"%optimizer
            optimizer = optimizers[optimizer.lower()]
            print("Modulator %s initialized with %s optimizer."%(self.model.name, optimizer.__name__))
            self.param_dicts = [\
                    {'params': self.model.mu_parameters(), 'lr':stepsize_mu},
                    {'params': self.std, 'lr':stepsize_sigma}]
            self.optimizer = optimizer(self.param_dicts)
        else:
            print("Modulator %s initialized WITHOUT an optimizer"%(self.model.name))
            self.optimizer = None
            if hasattr(self.model, "mu_parameters"):
                self.param_dicts = [\
                    {'params': self.model.mu_parameters(), 'lr':stepsize_mu},
                    {'params': self.std, 'lr':stepsize_sigma}]
            else:
                self.param_dicts = []


    #input bit symbols
    #returns np cartesian or np complex
    def modulate(self, symbols:np.ndarray, dtype:str='complex', mode:str='exploit') -> np.ndarray:
        symbols = torch.from_numpy(symbols).float()
        if mode == 'explore' and not self.model.name.lower() == 'classic':
            means = self.model(symbols)
            std_bounded = nn.functional.relu(self.std-self.std_min)+self.std_min
            std_bounded = self.std_max-nn.functional.relu(-(std_bounded-self.std_max))
            self.re_normal = torch.distributions.normal.Normal(means[:,0], std_bounded[0])
            self.im_normal = torch.distributions.normal.Normal(means[:,1], std_bounded[1])
            cartesian_points = torch.stack((self.re_normal.sample(), self.im_normal.sample()),1)
        elif self.model.name.lower() == 'classic' or mode == 'exploit':
            cartesian_points = self.model(symbols)
        else:
            raise Exception("modulator.modulate mode=%s not recognized; accepted inputs: 'explore', 'exploit'"%mode)
        if dtype == 'cartesian':
            return cartesian_points.detach().numpy()
        elif dtype == 'complex':
            return cartesian_2d_to_complex(cartesian_points.detach().numpy().astype(np.complex64))
        else:
            raise Exception("modulator.modulate dtype=%s not recognized; accepted inputs: 'cartesian', 'complex'"%dtype)

    #input torch.Tensor of bit symbols
    #returns torch.Tensor of cartesian points
    def modulate_tensor(self, symbols:torch.Tensor, mode:str='exploit') -> torch.Tensor:
        if mode == 'explore' and not self.model.name.lower() == 'classic':
            means = self.model(symbols)
            std_bounded = nn.functional.relu(self.std-self.std_min)+self.std_min
            std_bounded = self.std_max-nn.functional.relu(-(self.std_bounded-self.std_max))
            self.re_normal = torch.distributions.normal.Normal(means[:,0], self.std_bounded[0])
            self.im_normal = torch.distributions.normal.Normal(means[:,1], self.std_bounded[1])
            cartesian_points = torch.stack((self.re_normal.sample(), self.im_normal.sample()),1)
        elif self.model.name.lower() == 'classic' or mode == 'exploit':
            cartesian_points = self.model(symbols)
        else:
            raise Exception("modulator.modulate_tensor mode=%s not recognized accepted inputs: 'explore', 'exploit'"%mode)
        return cartesian_points

    #input bit symbols, cartesian/complex actions, bit symbols
    def update(self, symbols:np.ndarray, actions:np.ndarray, received_symbols:np.ndarray, **kwargs):
        if hasattr(self.model, "update"):
            kwargs['symbols'] = symbols
            kwargs['actions'] = actions
            kwargs['received_symbols'] = received_symbols
            self.model.update(**kwargs)
            return 
        else:
            assert self.optimizer, "Modulator is not initialized with an optimizer"
            if len(actions.shape)==2:
                cartesian_actions = torch.from_numpy(actions).float()
            elif len(actions.shape)==1:
                cartesian_actions = torch.from_numpy(np.stack((actions.real.astype(np.float32), actions.imag.astype(np.float32)), axis=-1))
            reward = torch.from_numpy(-np.sum(symbols ^ received_symbols, axis=1)).float() #correct bits = 0, incorrect bits = -1 #TESTED
            #reward = np.sum(1 - 2 * (symbols ^ received_symbols), axis=1) #correct bits = 1, incorrect bits = -1 #NOT TESTED
            log_probs =  self.re_normal.log_prob(cartesian_actions[:,0]) + self.im_normal.log_prob(cartesian_actions[:,1])
            baseline = torch.mean(reward)
            loss = -torch.mean(log_probs * (reward - self.lambda_baseline * baseline))
            if self.lambda_center > 0:
                loss +=  self.lambda_center * self.model.location_loss()
            if self.lambda_l1 > 0:
                loss +=  self.lambda_l1 * self.model.l1_loss()
            if self.lambda_l2 > 0:
                loss +=  self.lambda_l2 * self.model.l2_loss()
            
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            return -np.average(reward)

    def get_std(self):
        if hasattr(self.model, 'mu_parameters'):
            return self.std.data.detach().numpy()
        else:
            return [0.0 , 0.0]

    def get_constellation(self):
        all_unique_symbols = self.all_symbols
        cartesian_constellation = self.modulate(all_unique_symbols, dtype='cartesian')
        return cartesian_constellation

    def get_param_dicts(self):
        return self.param_dicts

    def get_regularization_loss(self):
        r_loss = torch.tensor(0.0).float()
        if not hasattr(self.model, "update"):
            if self.lambda_center > 0:
                r_loss +=  self.lambda_center * self.model.location_loss()
            if self.lambda_l1 > 0:
                r_loss +=  self.lambda_l1 * self.model.l1_loss()
            if self.lambda_l2 > 0:
                r_loss +=  self.lambda_l2 * self.model.l2_loss()
        return r_loss
