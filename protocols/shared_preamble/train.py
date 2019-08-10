###SHARED PREAMBLE###
from utils.util_data import integers_to_symbols
from utils.util_data import add_cartesian_awgn as add_awgn
from protocols.roundtrip_evaluate import roundtrip_evaluate as evaluate
from typing import List
import numpy as np
from utils.util_lookup_table import load_lookup_table, BER_lookup_table


###SHARED PREAMBLE###
def train(*,
          agents,
          bits_per_symbol: int,
          batch_size: int,
          num_iterations: int,
          results_every: int,
          train_SNR_db: float,
          signal_power: float,
          early_stopping_db_off: float = 1.0,
          plot_callback,
          **kwargs
          ):

    br = BER_lookup_table()
    integers_to_symbols_map = integers_to_symbols(np.arange(0, 2 ** bits_per_symbol), bits_per_symbol)

    if kwargs['verbose']:
        print("shared_preamble train.py")

    Amod = agents[0].mod
    Ademod = agents[0].demod
    Bmod = agents[1].mod
    Bdemod = agents[1].demod

    prev_preamble = None
    prev_actions = None
    batches_sent_roundtrip = 0
    results = []
    for i in range(num_iterations + 1):
        # A.mod(preamble) |               | B.demod(signal forward)      |==> B update demod     |
        #                 |--> channel -->|                              |                       |--> switch (A,B = B,A)
        # A.mod(pre-half) |               | A.demod(signal backward)     |==> B update mod       |
        integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[batch_size])
        preamble = integers_to_symbols_map[integers]  # new shared preamble
        # A
        if prev_preamble is not None:
            c_signal_backward = Amod.modulate(preamble_halftrip, mode='explore', dtype='cartesian')
        c_signal_forward = Amod.modulate(preamble, mode='explore', dtype='cartesian')

        # Channel
        if prev_preamble is not None:
            c_signal_backward_noisy = add_awgn(c_signal_backward, SNR_db=train_SNR_db, signal_power=signal_power)
        c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=train_SNR_db, signal_power=signal_power)

        if prev_preamble is not None:
            preamble_roundtrip = Bdemod.demodulate(c_signal_backward_noisy)
            # Update mod after a roundtrip pass
            Bmod.update(prev_preamble, prev_actions, preamble_roundtrip)
            batches_sent_roundtrip += 1

        # guess of new preamble
        preamble_halftrip = Bdemod.demodulate(c_signal_forward_noisy)
        # Update demod after a oneway pass
        if i < num_iterations:
            # the last iteration is just to complete the roundtrip, do not update halftrip
            Bdemod.update(c_signal_forward_noisy, preamble)

        prev_preamble, prev_actions = preamble, c_signal_forward

        # SWITCH
        Amod, Ademod, Bmod, Bdemod = Bmod, Bdemod, Amod, Ademod

        ############### STATS ##########################
        if i % results_every == 0 or i == num_iterations:
            new_kwargs = {**kwargs,
                          'protocol': "shared_preamble",
                          # 'agents': agents,
                          'bits_per_symbol': bits_per_symbol,
                          'train_SNR_db': train_SNR_db,
                          'signal_power': signal_power,
                          'num_iterations': num_iterations,
                          'results_every': results_every, 'iteration': i,
                          }
            result = evaluate(agent1=agents[0], agent2=agents[1], **new_kwargs)
            result['train_SNR_db'] = train_SNR_db
            result['iteration'] = i
            result['roundtrip_batches'] = batches_sent_roundtrip
            result['batch_size'] = batch_size

            test_snr_dbs = result.get('test_SNR_dbs', None)
            if test_snr_dbs is not None:
                test_bers = result['test_bers']
                db_off_for_each_test_snr = list(map(lambda test:
                                            test[0] - br.get_optimal_SNR_for_BER_roundtrip(test[1], bits_per_symbol),
                                            zip(test_snr_dbs, test_bers)))
                result['db_off'] = db_off_for_each_test_snr
            results += [result]
            try:
                if all(np.array(db_off_for_each_test_snr) <= early_stopping_db_off):
                    print("STOPPED AT ITERATION: %i"%i)
                    print(['training SNR', '0 BER', '1e-5 BER', '1e-4 BER', '1e-3 BER', '1e-2 BER', '1e-1 BER'])
                    print("TEST SNR dBs : ", test_snr_dbs)
                    print("dB off Optimal : ", db_off_for_each_test_snr)
                    print("Early Stopping dBs off: %d"%early_stopping_db_off)
                    return results
            except NameError:
                continue

            # plot_callback(new_kwargs)
    return results
