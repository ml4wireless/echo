from collections import OrderedDict
import torch
import numpy as np
import threading
import queue

from utils.util_data import integers_to_symbols
from models.modulator import Modulator
from models.modulator_models.Classic import Classic as ClassicMod
from models.demodulator import Demodulator
from models.demodulator_models.Classic import Classic as ClassicDemod


def update_parameters(model, loss, params=None, step_size=0.5, first_order=False):
    """Update of the meta-parameters with one step of gradient descent on the
    loss function.
    Parameters
    ----------
    model : The model. Meta-parameters should be named.
    loss : `torch.Tensor` instance
        The value of the inner-loss. This is the result of the training dataset
        through the loss function.
    params : `collections.OrderedDict` instance, optional
        Dictionary containing the meta-parameters of the model. If `None`, then
        the values stored in `model.named_parameters()` are used. This is
        useful for running multiple steps of gradient descent as the inner-loop.
    step_size : int, `torch.Tensor`, or `collections.OrderedDict` instance (default: 0.5)
        The step size in the gradient update. If an `OrderedDict`, then the
        keys must match the keys in `params`.
    first_order : bool (default: `False`)
        If `True`, then the first order approximation of MAML is used.
    """
    if params is None:
        params = model.meta_parameters()

    grads = torch.autograd.grad(loss, params.values(),
                                create_graph=not first_order)

    out = OrderedDict()

    if isinstance(step_size, (dict, OrderedDict)):
        for (name, param), grad in zip(params.items(), grads):
            out[name] = param - step_size[name] * grad
    else:
        for (name, param), grad in zip(params.items(), grads):
            out[name] = param - step_size * grad

    return out


class ClassicModDataset(torch.utils.data.Dataset):
    """
    A Dataset which returns meta-batches of symbols and their rotated classic modulations
    """
    def __init__(self, *, bits_per_symbol: int, ntasks: int = 16, length: int = int(1e9)):
        super(ClassicModDataset, self).__init__()
        self.bits_per_symbol = bits_per_symbol
        self.ntasks = ntasks
        self.length = length

    def __getitem__(self, _):
        mods = [Modulator(model=ClassicMod, bits_per_symbol=self.bits_per_symbol,
                          rotation=rot, verbose=False)
                for rot in np.random.uniform(0, 2 * np.pi, self.ntasks)]
        return mods

    def __len__(self):
        return self.length


class ClassicModDatasetThreaded:
    """
    A threaded Dataset which returns meta-batches of symbols and their rotated classic modulations
    """
    def __init__(self, *, bits_per_symbol: int, ntasks: int = 16, length: int = int(1e9)):
        self.dataset = ClassicModDataset(bits_per_symbol=bits_per_symbol, ntasks=ntasks, length=length)
        self.bits_per_symbol = bits_per_symbol
        self.length = length
        self.queue = queue.Queue(maxsize=8)
        self.thread = threading.Thread(target=ClassicModDatasetThreaded.get_batches, args=(self,))
        self.thread.daemon = True
        self.thread.start()

    def __getitem__(self, _):
        data = self.queue.get(block=True)
        return data

    def __len__(self):
        return self.length

    def get_batches(self):
        while True:
            data = self.dataset[None]
            self.queue.put(data, block=True)


class ClassicDemodDataset(torch.utils.data.Dataset):
    """
    A Dataset which returns meta-batches rotated classic demodulators
    """
    def __init__(self, *, bits_per_symbol: int, ntasks: int = 16, length: int = int(1e9)):
        super(ClassicDemodDataset, self).__init__()
        self.bits_per_symbol = bits_per_symbol
        self.ntasks = ntasks
        self.length = length

    def __getitem__(self, _):
        demods = [Demodulator(model=ClassicDemod, bits_per_symbol=self.bits_per_symbol,
                              rotation=rot, verbose=False)
                  for rot in np.random.uniform(0, 2 * np.pi, self.ntasks)]
        return demods

    def __len__(self):
        return self.length


class ClassicDemodDatasetThreaded:
    """
    A threaded Dataset which returns meta-batches of rotated classic demodulators
    """
    def __init__(self, *, bits_per_symbol: int, ntasks: int = 16, length: int = int(1e9)):
        self.dataset = ClassicDemodDataset(bits_per_symbol=bits_per_symbol, ntasks=ntasks, length=length)
        self.bits_per_symbol = bits_per_symbol
        self.length = length
        self.queue = queue.Queue(maxsize=4)
        self.thread = threading.Thread(target=ClassicDemodDatasetThreaded.get_batches, args=(self,))
        self.thread.daemon = True
        self.thread.start()

    def __getitem__(self, _):
        data = self.queue.get(block=True)
        return data

    def __len__(self):
        return self.length

    def get_batches(self):
        while True:
            data = self.dataset[None]
            self.queue.put(data, block=True)

