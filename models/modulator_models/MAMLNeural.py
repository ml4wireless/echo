from typing import Optional

import torch

from models.modulator_models.Neural import Neural
from collections import OrderedDict

import torch.nn.functional as F


class MAMLNeural(Neural):
    def __init__(self, *,
                 activation_fn_hidden: str = 'tanh',
                 activation_fn_output: Optional[str] = None,
                 **kwargs):
        super(MAMLNeural, self).__init__(activation_fn_hidden=activation_fn_hidden,
                                         activation_fn_output=activation_fn_output,
                                         **kwargs)
        activation_functionals = {
            'lrelu': F.leaky_relu,
            'relu': F.relu,
            'tanh': torch.tanh,
            'sigmoid': F.sigmoid
        }
        self.activation_fn_hidden = activation_functionals[activation_fn_hidden]
        self.activation_fn_output = None
        if activation_fn_output:
            self.activation_fn_output = activation_functionals[activation_fn_output]

    def meta_parameters(self, recurse=True):
        return OrderedDict(self.base.named_parameters())

    def _forward_functional(self, input: torch.Tensor, params):
        out = input.float()
        for i in range(len(self.hidden_layers)):
            out = F.linear(out,
                           weight=params['linear{0}.weight'.format(i)],
                           bias=params['linear{0}.bias'.format(i)])
            out = self.activation_fn_hidden(out)
        out = F.linear(out,
                       weight=params['linear{0}.weight'.format(len(self.hidden_layers))],
                       bias=params['linear{0}.bias'.format(len(self.hidden_layers))])
        if self.activation_fn_output:
            out = self.activation_fn_output(out)
        return out

    def forward(self, input: torch.Tensor, params=None) -> torch.Tensor:
        """
        We have to apply functional operations here to get the
        gradients we need later.
        """
        assert len(input.shape) == 2  # input shape should be [N_symbols, bits_per_symbol]
        assert input.shape[1] == self.bits_per_symbol
        if params is None:
            params = self.meta_parameters()
        means = self._forward_functional(input, params)
#         print(self.restrict_energy)
        if (self.restrict_energy == 1):
            means = self.normalize_constellation(means, params)
        elif (self.restrict_energy == 2):
            means = self.normalize_symbols(means)
        elif (self.restrict_energy == 3):
            means = self.center_and_normalize_constellation(means, params)
        return means

    def location_loss(self, params=None) -> torch.Tensor:
        return torch.norm(torch.mean(self.forward(self.all_unique_symbols, params))) ** 2

    def set_parameters(self, param_dict: OrderedDict):
        self.base.load_state_dict(param_dict)

    def normalize_constellation(self, means, params):
        # Get average power
        # WARNING: this can cause memory and speed issues for higher modulation orders like QAM 64000
        avg_power = torch.mean(torch.sum((self._forward_functional(self.all_unique_symbols, params)) ** 2, dim=-1))
        # Get normalization factor based on maximum constraint of 1 on average power
        if self.max_amplitude > 0:
            normalization_factor = torch.sqrt((torch.relu(avg_power - self.max_amplitude) + self.max_amplitude) / self.max_amplitude)
        else:
            normalization_factor = torch.sqrt(torch.relu(avg_power - 1.0) + 1.0)
        # Divide by normalization factor to get modulated symbols
        means = means / normalization_factor
        return means

    def center_and_normalize_constellation(self, means, params):
        const_means = self.center_means(self._forward_functional(self.all_unique_symbols, params))
        avg_power = torch.mean(torch.sum(const_means ** 2, dim=-1))
        # Get normalization factor based on maximum constraint of 1 on average power
        if self.max_amplitude > 0:
            normalization_factor = torch.sqrt((torch.relu(avg_power - self.max_amplitude) + self.max_amplitude) / self.max_amplitude)
        else:
            normalization_factor = torch.sqrt(torch.relu(avg_power - 1.0) + 1.0)
        # Divide by normalization factor to get modulated symbols
        means = self.center_means(means) / normalization_factor
        return means
