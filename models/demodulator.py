from typing import Optional

import numpy as np
import torch
from torch import nn

from utils.util_data import integers_to_symbols, symbols_to_integers


class Demodulator():
    def __init__(self, *,
                 model,
                 bits_per_symbol,
                 optimizer: Optional[str] = 'adam',
                 stepsize_cross_entropy: float = 1e-3,  #
                 cross_entropy_weight: float = 1.0,  #
                 epochs: int = 5,
                 lambda_l1: float = 0.0,
                 lambda_l2: float = 0.0,
                 device: str = 'cpu',
                 **kwargs
                 ):
        self.device = torch.device(device)
        self.epochs = epochs
        self.bits_per_symbol = bits_per_symbol
        self.model = model(bits_per_symbol=bits_per_symbol, device=device,
                           **kwargs)
        self.name = self.model.name
        self.lambda_l1 = torch.tensor(lambda_l1, device=self.device).float()
        self.lambda_l2 = torch.tensor(lambda_l2, device=self.device).float()
        self.stepsize_cross_entropy = torch.tensor(stepsize_cross_entropy).float().to(self.device)
        self.integers_to_symbols_map = integers_to_symbols(np.arange(0, 2 ** bits_per_symbol), bits_per_symbol)

        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }

        if optimizer and hasattr(self.model, 'parameters') and not hasattr(self.model, "update"):
            assert optimizer.lower() in optimizers.keys(), "demodulator optimizer=%s not supported" % optimizer
            optimizer = optimizers[optimizer.lower()]
            if kwargs['verbose']:
                print("Demodulator %s initialized with %s optimizer." % (self.model.name, optimizer.__name__))
            self.cross_entropy_weight = torch.tensor(cross_entropy_weight, device=self.device).float()
            self.param_dicts = [
                {'params': self.model.parameters(), 'lr': stepsize_cross_entropy},
            ]
            self.optimizer = optimizer(self.param_dicts, lr=stepsize_cross_entropy)
        else:
            if kwargs['verbose']:
                print("Demodulator %s initialized WITHOUT an optimizer" % (self.model.name))
            self.optimizer = None
            if hasattr(self.model, 'parameters'):
                self.param_dicts = [
                    {'params': self.model.parameters(), 'lr': stepsize_cross_entropy},
                ]
            else:
                self.param_dicts = []

    # input cartesian/complex np.ndarray
    # output bit symbols np.ndarray
    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            if len(signal.shape) == 2:
                cartesian_points = torch.from_numpy(signal).float().to(self.device)
            elif len(signal.shape) == 1:
                cartesian_points = torch.from_numpy(np.stack((signal.real, signal.imag), axis=-1)).float().to(self.device)
            logits = self.model.forward(cartesian_points)
            _, actions = torch.max(logits, 1)  # actions are class indexes
            symbols = self.integers_to_symbols_map[actions.detach().cpu().numpy().astype(int)]
        return symbols

    # input cartesian torch.Tensor
    # output logits for symbols torch.Tensor, map of class id (int) to the bit symbol np.ndarray
    def demodulate_tensor(self, signal: torch.Tensor) -> torch.Tensor:
        cartesian_points = signal
        logits = self.model.forward(cartesian_points)
        return logits

    def demodulate_tensor_no_grad(self, signal: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cartesian_points = signal
            logits = self.model.forward(cartesian_points)
        return logits

    # input cartesian/complex np.ndarray, bit symbols np.ndarray
    def update(self, signal: np.ndarray, true_symbols: np.ndarray, **kwargs):
        model = self.model
        if hasattr(model, "update"):
            kwargs['signal'] = signal
            kwargs['true_symbols'] = true_symbols
            model.update(**kwargs)
            return
        else:
            assert self.optimizer, "Demodulator is not initialized with an optimizer"
            # train
            if len(signal.shape) == 2:
                cartesian_points = torch.from_numpy(signal).float().to(self.device)
            elif len(signal.shape) == 1:
                cartesian_points = torch.from_numpy(
                    np.stack((signal.real.astype(np.float32), signal.imag.astype(np.float32)), axis=-1)).to(self.device)
            l1, l2, cross_entropy_weight, optimizer = [self.lambda_l1, self.lambda_l2,
                                                       self.cross_entropy_weight, self.optimizer]
            criterion = nn.CrossEntropyLoss()
            for _ in range(self.epochs):
                logits = model.forward(cartesian_points)
                target = torch.from_numpy(symbols_to_integers(true_symbols)).to(self.device)
                loss = cross_entropy_weight * torch.mean(criterion(logits, target))
                if l1 > 0:
                    loss += l1 * model.l1_loss()
                if l2 > 0:
                    loss += l2 * model.l2_loss()
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            return

    # input cartesian/complex np.ndarray
    # output class (int) np.ndarray
    def get_demod_grid(self, grid_2d: np.ndarray):
        symbols = self.demodulate(grid_2d)
        classes = symbols_to_integers(symbols)
        return classes

    def get_param_dicts(self):
        return self.param_dicts

    def get_regularization_loss(self):
        r_loss = torch.tensor(0.0).float().to(self.device)
        if not hasattr(self.model, "update"):
            if self.lambda_l1 > 0:
                r_loss += self.lambda_l1 * self.model.l1_loss()
            if self.lambda_l2 > 0:
                r_loss += self.lambda_l2 * self.model.l2_loss()
        return r_loss

    def get_weights(self):
        assert hasattr(self.model, 'parameters'), "Can't get weights for a parameterless model"
        return {'demod': self.model.base.state_dict()}

    def load_weights_file(self, weight_file):
        assert hasattr(self.model, 'parameters'), "Can't load weights for a parameterless model"
        self.set_weights(torch.load(weight_file, map_location=self.device))

    def set_weights(self, weight_dict):
        assert hasattr(self.model, 'parameters'), "Can't set weights for a parameterless model"
        self.model.base.load_state_dict(weight_dict['demod'])
