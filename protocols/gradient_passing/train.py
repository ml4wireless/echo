###GRADIENT PASSING###
from utils.util_data import integers_to_symbols, symbols_to_integers
from utils.util_data import add_cartesian_awgn as add_awgn
import torch
from typing import List
import numpy as np


def get_random_preamble(n, bits_per_symbol):
    integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[n])
    return integers_to_symbols(integers, bits_per_symbol)


###GRADIENT PASSING###
def train(*,
          agents,
          optimizer,
          agent2=None,
          bits_per_symbol: int,
          batch_size: int,
          num_iterations: int,
          results_every: int,
          SNR_db: float,
          test_batch_size=int,
          signal_power=float,
          plot_callback,
          evaluate_callback, **kwargs
          ):
    if kwargs['verbose']:
        print("gradient_passing train.py")

    A = agents[0]
    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
    }

    if optimizer:
        assert optimizer.lower() in optimizers.keys(), "modulator optimizer=%s not supported" % optimizer
        optimizer = optimizers[optimizer.lower()]
        print("gradient_passing initialized with %s optimizer." % (optimizer.__name__))
        optimizer = optimizer(A.mod.get_param_dicts() + A.demod.get_param_dicts())
    else:
        print("gradient_passing initialized WITHOUT an optimizer")
        optimizer = None

    batches_sent = 0
    loss_criterion = torch.nn.CrossEntropyLoss()

    for i in range(num_iterations + 1):
        preamble_labels = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[batch_size])
        preamble = integers_to_symbols(preamble_labels, bits_per_symbol)
        preamble = torch.from_numpy(preamble).float()
        ##MODULATE/action
        c_signal_forward = A.mod.modulate_tensor(preamble)
        ##CHANNEL
        c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=SNR_db, signal_power=signal_power)
        ##DEMODULATE/update and pass loss to mod
        preamble_halftrip_logits = A.demod.demodulate_tensor(c_signal_forward_noisy)[0]
        loss = loss_criterion(input=preamble_halftrip_logits.float(), target=torch.from_numpy(preamble_labels))
        loss += A.mod.get_regularization_loss()
        loss += A.demod.get_regularization_loss()

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batches_sent += 1

        ############### STATS ##########################
        if i % results_every == 0 or i == num_iterations:
            kwargs = {
                'protocol': 'gradient_passing',
                'agents': agents,
                'bits_per_symbol': bits_per_symbol, 'SNR_db': SNR_db,
                'test_batch_size': test_batch_size, 'signal_power': signal_power,
                'num_iterations': num_iterations,
                'results_every': results_every, 'batch_size': batch_size,
                'batches_sent': batches_sent, 'iteration': i,
            }
            evaluate_callback(**kwargs)
            plot_callback(**kwargs)
