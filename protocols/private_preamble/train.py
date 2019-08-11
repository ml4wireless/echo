###PRIVATE PREAMBLE###
from utils.util_data import integers_to_symbols, add_cartesian_awgn as add_awgn
from utils.util_lookup_table import BER_lookup_table
from protocols.roundtrip_evaluate import roundtrip_evaluate as evaluate
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
          train_SNR_db: float,
          signal_power: float,
          early_stopping: bool = False,
          early_stopping_db_off: float = 1,
          verbose: bool = False,
          **kwargs,
          ):
    br = BER_lookup_table()
    early_stop = False
    integers_to_symbols_map = integers_to_symbols(np.arange(0, 2 ** bits_per_symbol), bits_per_symbol)
    if verbose:
        print("private_preamble train.py")

    Amod = agents[0].mod
    Ademod = agents[0].demod
    Bmod = agents[1].mod
    Bdemod = agents[1].demod
    prev_preamble = None
    prev_actions = None
    batches_sent = 0
    results = []
    for i in range(num_iterations + 1):
        # A.mod(preamble) |               | B.demod(signal forward)      |==> B has pre-half     |
        #                 |--> channel -->|                              |                       |--> switch (A,B = B,A)
        # A.mod(pre-half) |               | A.demod(signal backward)     |==> B update mod/demod |

        integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[batch_size])
        preamble = integers_to_symbols_map[integers]  # new private preamble
        # A
        if prev_preamble is not None:
            c_signal_backward = Amod.modulate(preamble_halftrip, mode='explore', dtype='cartesian')
        c_signal_forward = Amod.modulate(preamble, mode='explore', dtype='cartesian')

        # Channel
        if prev_preamble is not None:
            c_signal_backward_noisy = add_awgn(c_signal_backward, SNR_db=train_SNR_db, signal_power=signal_power)
        c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=train_SNR_db, signal_power=signal_power)

        if prev_preamble is not None:
            # Update mod and demod after a roundtrip pass
            Bdemod.update(c_signal_backward_noisy, preamble)
            preamble_roundtrip = Bdemod.demodulate(c_signal_backward_noisy)
            Bmod.update(prev_preamble, prev_actions, preamble_roundtrip)
            batches_sent += 2

        # guess of new preamble
        preamble_halftrip = Bdemod.demodulate(c_signal_forward_noisy)

        prev_preamble, prev_actions = preamble, c_signal_forward

        # SWITCH
        Amod, Ademod, Bmod, Bdemod = Bmod, Bdemod, Amod, Ademod

        ############### STATS ##########################
        if i % results_every == 0 or i == num_iterations:
            if verbose:
                print("ITER %i: Train SNR_db:% 5.1f" % (i, train_SNR_db))

            result = evaluate(agent1=agents[0],
                              agent2=agents[1],
                              bits_per_symbol=bits_per_symbol,
                              signal_power=signal_power,
                              verbose=verbose or i == num_iterations,
                              total_iterations=num_iterations // results_every,
                              completed_iterations=i // results_every,
                              **kwargs)

            test_SNR_dbs = result['test_SNR_dbs']
            test_bers = result['test_bers']
            db_off_for_test_snr = [testSNR - br.get_optimal_SNR_for_BER_roundtrip(testBER, bits_per_symbol)
                                   for testSNR, testBER in zip(test_SNR_dbs, test_bers)]
            ###ADD TO RESULT
            result['batches_sent'] = batches_sent
            result['db_off'] = db_off_for_test_snr
            results += [result]
            if early_stopping and all(np.array(db_off_for_test_snr) <= early_stopping_db_off):
                print("STOPPED AT ITERATION: %i" % i)
                print(['0 BER', '1e-5 BER', '1e-4 BER', '1e-3 BER', '1e-2 BER', '1e-1 BER'])
                print("TEST SNR dBs : ", test_SNR_dbs)
                print("dB off Optimal : ", db_off_for_test_snr)
                print("Early Stopping dBs off: %d" % early_stopping_db_off)
                early_stop = True
                break
    info = {
        'bits_per_symbol': bits_per_symbol,
        'train_SNR_db': train_SNR_db,
        'num_results': len(results),
        'early_stop': early_stop,
        'early_stop_threshold_db_off': early_stopping_db_off,
        'batch_size': batch_size,
        'num_agents': 2,
    }
    return info, results
