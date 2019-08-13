from itertools import product
from typing import Union

import numpy as np
import torch
from torch import nn

from utils.util_data import get_all_unique_symbols


class Poly(nn.Module):
    def __init__(self, *,
                 bits_per_symbol: Union[int, float],
                 restrict_energy: Union[int, float] = 1,
                 max_amplitude: float = 0.0,
                 **kwargs,
                 ):

        super(Poly, self).__init__()
        self.name = 'poly'
        self.model_type = 'modulator'
        self.linear = nn.Linear(2 ** bits_per_symbol, 2, bias=False)
        def _init_weights(m):
            if type(m) == nn.Linear:
                y = 1.0 / np.sqrt(m.in_features)
                m.weight.data.uniform_(-y, y)

        self.linear.apply(_init_weights)
        ####################
        # Class Variables
        ####################
        self.bits_per_symbol = bits_per_symbol
        self.restrict_energy = restrict_energy
        self.max_amplitude = max_amplitude
        self.all_unique_symbols = torch.tensor(get_all_unique_symbols(bits_per_symbol=bits_per_symbol)).float()
        # self.exp is used to build polynomial features of the symbols i.e. [b1, b2] ---> [b1, b2, b1b2, 1].
        # We ignore b1**2 and b2**2 and higher order polynomial features because bits are 0 or 1
        # build_exp is defined below
        self.exp = torch.from_numpy(np.array(list(product([0, 1], repeat=bits_per_symbol)))).float()
        self.poly_unique_symbols = None

    def polynomial(self, input):
        return torch.prod(torch.pow(input[:, None, :], self.exp[None, ...]), 2)
        # return torch.stack([torch.prod(torch.pow(input.float(), x), 1) for x in self.exp]).t()

    # HARDCODED SOLUTION TO QAM16 MODULATION
    def QAM16(self, input: torch.Tensor):
        coeff = torch.tensor([[-1, 0, 0, 0, 2 / 3, 0, 0, 0, 2, 0, 0, 0, -4 / 3, 0, 0, 0],
                              [-1, 2 / 3, 2, -4 / 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).float().t()
        # print(self.polynomial(input).shape, coeff.shape)
        means = torch.mm(self.polynomial(input), coeff)
        return means

    def normalize_constellation(self, means: torch.Tensor) -> torch.Tensor:
        # Get average power of the constellation
        # WARNING: this can cause memory and speed issues for higher modulation orders like QAM 64000
        if self.poly_unique_symbols is None:
            self.poly_unique_symbols = self.polynomial(self.all_unique_symbols)
        avg_power = torch.mean(torch.sum((self.linear(self.poly_unique_symbols)) ** 2, dim=-1))
        # Get normalization factor based on maximum constraint of self.max_amplitude on average power
        if self.max_amplitude > 0:
            normalization_factor = torch.sqrt(
                (torch.relu(avg_power - self.max_amplitude) + self.max_amplitude) / self.max_amplitude)
        else:
            normalization_factor = torch.sqrt(torch.relu(avg_power - 1.0) + 1.0)
            # Divide by normalization factor to get modulated symbols
        return means / normalization_factor

    def normalize_symbols(self, means: torch.Tensor) -> torch.Tensor:
        # Get average energy of the batch
        avg_power = torch.sqrt(torch.mean(torch.sum(means ** 2, dim=1)))
        normalization = torch.nn.functional.relu(avg_power - 1.0) + 1.0
        means = means / normalization
        return means

    def center_and_normalize_constellation(self, means: torch.Tensor) -> torch.Tensor:
        if self.poly_unique_symbols is None:
            self.poly_unique_symbols = self.polynomial(self.all_unique_symbols)
        avg_power = torch.mean(torch.sum(self.center_means(self.linear(self.poly_unique_symbols)) ** 2, dim=-1))
        normalization = torch.nn.functional.relu(avg_power - 1) + 1.0
        center = means.mean(dim=0)
        centered = means - center
        means = centered / normalization
        return means

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        means = self.linear(self.polynomial(input))
        # means = self.QAM16(input) #ANSWER FOR QAM16
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

    def l1_loss(self) -> torch.Tensor:
        return torch.norm(self.linear.weight, p=1)

    def l2_loss(self) -> torch.Tensor:
        return torch.norm(self.linear.weight, p=2)

    def location_loss(self) -> torch.Tensor:
        return torch.norm(torch.mean(self.forward(self.all_unique_symbols))) ** 2

    def mu_parameters(self) -> torch.Tensor:
        return self.linear.parameters()
