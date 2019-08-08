###PRIVATE PREAMBLE###
from utils.util_data import integers_to_symbols
from utils.util_data import add_complex_awgn as add_awgn
from typing import List
import numpy as np


def get_random_preamble(n, bits_per_symbol):
    integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[n])
    return integers_to_symbols(integers, bits_per_symbol)


###PRIVATE PREAMBLE###
def train(*,
          agents,
          bits_per_symbol: int,
          batch_size: int,
          num_iterations: int,
          results_every: int,
          SNR_db: float,
          signal_power=float,
          plot_callback,
          evaluate_callback,
          **kwargs,
          ):
    if kwargs['verbose']:
        print("private_preamble train.py")
    A = agents[0]
    B = agents[1]

    prev_preamble = None
    prev_actions = None
    batches_sent_roundtrip = 0
    for i in range(num_iterations + 1):
        # A.mod(preamble) |               | B.demod(signal forward)      |==> B has pre-half     |
        #                 |--> channel -->|                              |                       |--> switch (A,B = B,A)
        # A.mod(pre-half) |               | A.demod(signal backward)     |==> B update mod/demod |

        preamble = get_random_preamble(batch_size, bits_per_symbol)  # new shared preamble
        # A
        if prev_preamble is not None:
            c_signal_backward = A.mod.modulate(preamble_halftrip, mode='explore')
        c_signal_forward = A.mod.modulate(preamble, mode='explore', dtype='complex')

        # Channel
        if prev_preamble is not None:
            c_signal_backward_noisy = add_awgn(c_signal_backward, SNR_db=SNR_db, signal_power=signal_power)
        c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=SNR_db, signal_power=signal_power)

        if prev_preamble is not None:
            # Update mod and demod after a roundtrip pass
            B.demod.update(c_signal_backward_noisy, preamble)
            preamble_roundtrip = B.demod.demodulate(c_signal_backward_noisy)
            B.mod.update(prev_preamble, prev_actions, preamble_roundtrip)
            batches_sent_roundtrip += 1

            # guess of new preamble
        preamble_halftrip = B.demod.demodulate(c_signal_forward_noisy)

        prev_preamble, prev_actions = preamble, c_signal_forward
        # SWITCH
        A, B = B, A

        ############### STATS ##########################
        if i % results_every == 0 or i == num_iterations:
            new_kwargs = {**kwargs,
                          'protocol': "shared_preamble",
                          'agents': agents,
                          'bits_per_symbol': bits_per_symbol,
                          'SNR_db': SNR_db,
                          'signal_power': signal_power,
                          'num_iterations': num_iterations,
                          'results_every': results_every, 'batch_size': batch_size,
                          'batches_sent_roundtrip': batches_sent_roundtrip, 'iteration': i,
                          }
            evaluate_callback(**new_kwargs)
            # plot_callback(**new_kwargs)
