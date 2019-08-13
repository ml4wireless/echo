###PRIVATE PREAMBLE###
import numpy as np

from utils.util_data import integers_to_symbols, add_cartesian_awgn as add_awgn


###PRIVATE PREAMBLE###
def trainer(*,
          agents,
          bits_per_symbol: int,
          batch_size: int,
          train_SNR_db: float,
          signal_power: float = 1.0,
          backwards_only: bool = False,
          **kwargs
          ):
    integers_to_symbols_map = integers_to_symbols(np.arange(0, 2 ** bits_per_symbol), bits_per_symbol)
    A = agents[0]
    B = agents[1]

    batches_sent = 0
    integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[batch_size])
    preamble = integers_to_symbols_map[integers]  # new shared preamble
    # A
    if A.to_echo is not None:
        c_signal_backward = A.mod.modulate(A.to_echo, mode='explore', dtype='cartesian')
    c_signal_forward = A.mod.modulate(preamble, mode='explore', dtype='cartesian')
    A.preamble = preamble
    A.actions = c_signal_forward

    # Channel
    if A.to_echo is not None:
        c_signal_backward_noisy = add_awgn(c_signal_backward, SNR_db=train_SNR_db, signal_power=signal_power)
    c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=train_SNR_db, signal_power=signal_power)

    # B
    if A.to_echo is not None:
        B.demod.update(c_signal_backward_noisy, B.preamble)
        preamble_roundtrip = B.demod.demodulate(c_signal_backward_noisy)
        # Update mod after a roundtrip pass
        B.mod.update(B.preamble, B.actions, preamble_roundtrip)
        batches_sent += 2

    # guess of new preamble
    B.to_echo = B.demod.demodulate(c_signal_forward_noisy)

    return A, B, batches_sent
