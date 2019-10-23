from typing import Union, Optional

import numpy as np
import torch
from torch import nn
from torch.distributions.normal import Normal

from utils.util_data import cartesian_2d_to_complex, integers_to_symbols


class Modulator():
    def __init__(self,*, 
                model,
                bits_per_symbol:Union[float,int],
                optimizer: Optional[str] = 'adam',
                stepsize_mu: float = 0.0,
                stepsize_sigma: float = 0.0,
                initial_std: float = 0.1,
                min_std: float = 1e-5,
                max_std: float = 1e2,
                lambda_baseline:float = 0.0, #used in update method
                lambda_center:float = 0.0,   #used in update method
                lambda_l1:float = 0.0,       #used in update method
                lambda_l2:float = 0.0,       #used in update method
                **kwargs):

        self.model = model(bits_per_symbol = bits_per_symbol, **kwargs)
        self.name = self.model.name
        self.bits_per_symbol = bits_per_symbol
        self.log_std_min=np.log(min_std) * torch.ones(2)
        self.log_std_max=np.log(max_std) * torch.ones(2)
        self.lambda_l1=torch.tensor(lambda_l1).float()
        self.lambda_l2=torch.tensor(lambda_l2).float()
        self.lambda_center=torch.tensor(lambda_center).float()
        self.lambda_baseline=torch.tensor(lambda_baseline).float()
        self.all_symbols = integers_to_symbols(np.arange(
            0, 2**bits_per_symbol), bits_per_symbol)
        self.log_std = nn.Parameter(np.log(initial_std) * torch.ones(2))
        # self.std.register_hook(lambda grad: print(grad))

        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }

        if optimizer and hasattr(self.model, "mu_parameters") and not hasattr(self.model, "update"):
            assert optimizer.lower() in optimizers.keys(), "modulator optimizer=%s not supported"%optimizer
            optimizer = optimizers[optimizer.lower()]
            if kwargs['verbose']:
                print("Modulator %s initialized with %s optimizer."%(self.model.name, optimizer.__name__))
            self.param_dicts = [\
                    {'params': self.model.mu_parameters(), 'lr':stepsize_mu},
                    {'params': self.log_std, 'lr':stepsize_sigma}]
            self.optimizer = optimizer(self.param_dicts)
        else:
            if kwargs['verbose']:
                print("Modulator %s initialized WITHOUT an optimizer"%(self.model.name))
            self.optimizer = None
            if hasattr(self.model, "mu_parameters"):
                self.param_dicts = [\
                    {'params': self.model.mu_parameters(), 'lr':stepsize_mu},
                    {'params': self.log_std, 'lr':stepsize_sigma}]
            else:
                self.param_dicts = []


    #input bit symbols
    #returns np cartesian or np complex
    def modulate(self, symbols:np.ndarray, dtype:str='complex', mode:str='exploit') -> np.ndarray:
        symbols = torch.from_numpy(symbols).float()
        if mode == 'explore' and not self.model.name.lower() == 'classic':
            means = self.model(symbols)
            # log_std = torch.min(torch.max(self.log_std_min, self.log_std), self.log_std_max)
            self.policy = Normal(means, self.log_std.exp())
            cartesian_points = self.policy.sample()
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
            # log_std = torch.min(torch.max(self.log_std_min, self.log_std), self.log_std_max)
            self.policy = Normal(means, self.log_std.exp())
            cartesian_points = self.policy.sample()
        elif self.model.name.lower() == 'classic' or mode == 'exploit':
            cartesian_points = self.model(symbols)
        else:
            raise Exception("modulator.modulate_tensor mode=%s not recognized accepted inputs: 'explore', 'exploit'"%mode)
        return cartesian_points

    #input bit symbols, cartesian/complex actions, bit symbols, regenerate Gaussian means
    #Vanilla PG, fine for simulation
    def update(self, symbols:np.ndarray, actions:np.ndarray, received_symbols:np.ndarray, rebuild_policy:bool=False, **kwargs):
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
            # reward =torch.from_numpy(np.sum(1 - 2 * (symbols ^ received_symbols), axis=1)).float() #correct bits = 1, incorrect bits = -1 #NOT TESTED
            if rebuild_policy:
                # Rebuild the policy here for the case where we overlap echo rounds
                tensor_symbols = torch.from_numpy(symbols).float()
                self.policy = Normal(self.model(tensor_symbols), self.log_std.exp())
            log_probs = self.policy.log_prob(cartesian_actions).sum(dim=1)
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
            # self.log_std = nn.functional.relu(self.log_std-self.log_std_min)+self.log_std_min
            # self.log_std = self.log_std_max-nn.functional.relu(-(self.log_std-self.log_std_max))
            return -np.average(reward)

    #input bit symbols, cartesian/complex actions, bit symbols
    #PPO instead of Vanilla Policy Gradient, Why? for efficiency when we are over the air... we can train more on the
    #single preamble we get. So sending a large preamble and training mini-batches over it.
    #lots of overhead to sending over air!
    def updatePPO(self, symbols: np.ndarray,
                        actions: np.ndarray,
                        received_symbols: np.ndarray,
                        clip_ratio = 0.2,
                        epochs = 5,
                        rebuild_policy = False, **kwargs):
        if hasattr(self.model, "update"):
            kwargs['symbols'] = symbols
            kwargs['actions'] = actions
            kwargs['received_symbols'] = received_symbols
            self.model.update(**kwargs)
            return
        else:
            assert self.optimizer, "Modulator is not initialized with an optimizer"
            if len(actions.shape) == 2:
                cartesian_actions = torch.from_numpy(actions).float()
            elif len(actions.shape) == 1:
                cartesian_actions = torch.from_numpy(
                    np.stack((actions.real.astype(np.float32), actions.imag.astype(np.float32)), axis=-1))
        if rebuild_policy:
            # Rebuild the policy here for the case where we overlap echo rounds
            tensor_symbols = torch.from_numpy(symbols).float()
            self.policy = Normal(self.model(tensor_symbols), self.log_std.exp())
        prev_log_prob = self.policy.log_prob(cartesian_actions).sum(dim=1).detach()
        reward = torch.from_numpy(-np.sum(symbols ^ received_symbols,
                                          axis=1)).float()  # correct bits = 0, incorrect bits = -1 #TESTED
        baseline = torch.mean(reward)
        adv = reward - self.lambda_baseline * baseline
        for i in range(epochs):
            log_prob = self.policy.log_prob(cartesian_actions).sum(dim=1)
            ratio = (log_prob - prev_log_prob).exp()
            min_adv = torch.where(adv > 0, (1 + clip_ratio) * adv,
                                  (1 - clip_ratio) * adv)
            loss = -(torch.min(ratio * adv, min_adv)).mean()
            if self.lambda_center > 0:
                loss += self.lambda_center * self.model.location_loss()
            if self.lambda_l1 > 0:
                loss += self.lambda_l1 * self.model.l1_loss()
            if self.lambda_l2 > 0:
                loss += self.lambda_l2 * self.model.l2_loss()

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

        # self.log_std = nn.functional.relu(self.log_std-self.log_std_min)+self.log_std_min
        # self.log_std = self.log_std_max-nn.functional.relu(-(self.log_std-self.log_std_max))
        return -np.average(reward)

    def get_std(self):
        if hasattr(self.model, 'mu_parameters'):
            return self.log_std.exp().data.detach().numpy()
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
