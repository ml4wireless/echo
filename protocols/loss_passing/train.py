###LOSS PASSING###
import numpy as np

from protocols.roundtrip_evaluate import roundtrip_evaluate as evaluate
from utils.util_data import integers_to_symbols, add_complex_awgn as add_awgn
from utils.util_lookup_table import BER_lookup_table


def get_random_preamble(n, bits_per_symbol):
    integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[n])
    return integers_to_symbols(integers, bits_per_symbol)


###LOSS PASSING###
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
          **kwargs
          ):
    br = BER_lookup_table()
    early_stop = False
    if verbose:
        print("loss_passing train.py")

    A = agents[0]
    batches_sent = 0
    results = []
    for i in range(num_iterations):

        preamble = get_random_preamble(batch_size, bits_per_symbol)
        ##MODULATE/action
        c_signal_forward = A.mod.modulate(preamble, mode='explore', dtype='complex')
        actions = c_signal_forward
        ##CHANNEL
        c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=train_SNR_db, signal_power=signal_power)
        ##DEMODULATE/update and pass loss to mod
        A.demod.update(c_signal_forward_noisy, preamble)
        preamble_halftrip = A.demod.demodulate(c_signal_forward_noisy)
        A.mod.update(preamble, actions, preamble_halftrip)
        batches_sent += 1

        ############### STATS ##########################
        if i % results_every == 0 or i == num_iterations-1:
            if verbose:
                print("ITER %i: Train SNR_db:% 5.1f" % (i, train_SNR_db))

            result = evaluate(agent1=agents[0],
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
            if early_stopping and  all(np.array(db_off_for_test_snr) <= early_stopping_db_off):
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
        'test_SNR_dbs': test_SNR_dbs,
        'early_stop': early_stop,
        'early_stop_threshold_db_off': early_stopping_db_off,
        'batch_size': batch_size,
        'num_agents': 1,
    }
    return info, results
