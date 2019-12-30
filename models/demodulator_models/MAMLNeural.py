from typing import Optional
import torch
import torch.nn.functional as F

from collections import OrderedDict

from models.demodulator_models.Neural import Neural


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

    def meta_parameters(self):
        return OrderedDict(self.base.named_parameters())

    def forward(self, input: torch.Tensor, params=None):
        """
        We have to apply functional operations here to get the
        gradients we need later.
        """
        assert len(input.shape) == 2, "input should be 2D cartesian points with shape [n_samples, 2]"
        if params is None:
            params = self.meta_parameters()
        out = input.float()
        for i in range(len(self.hidden_layers)):
            out = F.linear(out,
                    weight = params['linear{0}.weight'.format(i)],
                    bias = params['linear{0}.bias'.format(i)])
            out = self.activation_fn_hidden(out)
        out = F.linear(out,
                weight = params['linear{0}.weight'.format(len(self.hidden_layers))],
                bias = params['linear{0}.bias'.format(len(self.hidden_layers))])
        if self.activation_fn_output:
            out = self.activation_fn_output(out)
        return out

    def set_parameters(self, param_dict: OrderedDict):
        self.base.load_state_dict(param_dict)


