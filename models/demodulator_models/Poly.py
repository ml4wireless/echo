import math

import numpy as np
import torch
from torch import nn


def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


class Poly(nn.Module):
    def __init__(self,*, 
                bits_per_symbol, 
                degree_polynomial,
                batch_normalize=False,
                **kwargs): #initial_logstd, std_min, restrict_energy):
        super(Poly, self).__init__()
        self.name = 'polynomial'
        self.model_type = 'demodulator'
        self.batch_normalize = batch_normalize
        
        self.num_polynomial_terms =int( nCr(degree_polynomial + 2, degree_polynomial) )
        num_classes = 2**bits_per_symbol
        self.base = nn.Linear(self.num_polynomial_terms, num_classes, bias=False) #polynomial input
        def _init_weights(m):
            if type(m) == nn.Linear:
                y = 1.0/np.sqrt(m.in_features)
                m.weight.data.uniform_(-y, y)
        self.base.apply(_init_weights)

        # self.exp is the exponents used to builds polynomial features of the 2-D cartesian inputs
        self.exp = torch.from_numpy(
            np.array([[j, i - j] for i in range(degree_polynomial + 1) for j in range(i + 1)])).float()
        assert len(self.exp) == self.num_polynomial_terms, \
            "for each term (i.e. x^1*y^2) there should be a pair of exponents (i.e. [1,2]) in self.exp"

    # def polynomial(self, input): #map is slower for lower mod orders
    #     return torch.stack(list(map(lambda x: torch.prod(torch.pow(input.float(), x), 1), self.exp))).t()
    #     return torch.stack([torch.prod(torch.pow(input.float(), x), 1) for x in self.exp]).t()

    def polynomial(self, input):
        poly = torch.prod(torch.pow(input[:, None, :], self.exp[None, ...]), 2)
        # if self.batch_normalize:
        #     means = torch.mean(poly, 0, keepdim=True)
        #     std = torch.std(poly, 0, keepdim=True)
        #     return (poly - means) / std
        # else:
        return poly
   
    def forward(self, input:torch.Tensor):
        assert len(input.shape) == 2, "input should be 2D cartesian points with shape [n_samples, 2]"
        #2D cartesian input --> polynomial --> logit output (# classes = 2**bits_per_symbol)
        logits = self.base(self.polynomial(input))
        return logits
    
    def parameters(self):
        return self.base.parameters()

    def l1_loss(self):
        return torch.norm(self.base.weight, p=1)

    def l2_loss(self):
        return torch.norm(self.base.weight, p=2)