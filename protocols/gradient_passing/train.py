###GRADIENT PASSING###
from utils.util_data import integers_to_symbols, get_N0, get_awgn
from utils.util_lookup_table import BER_lookup_table
from protocols.roundtrip_evaluate import roundtrip_evaluate as evaluate
import torch
import numpy as np


def get_random_preamble(n, bits_per_symbol):
    integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[n])
    return integers_to_symbols(integers, bits_per_symbol)


###GRADIENT PASSING###
def train(*,
          agents,
          optimizer,
          bits_per_symbol: int,
          batch_size: int,
          num_iterations: int,
          results_every: int,
          train_SNR_db: float,
          signal_power: float,
          early_stopping: bool = False,
          early_stopping_db_off: float = .1,
          verbose: bool = False,
          **kwargs
          ):
    br = BER_lookup_table()
    early_stop = False
    if verbose:
        print("gradient_passing train.py")

    A = agents[0]
    optimizers = {
        'adam': torch.optim.Adam,
        'sgd': torch.optim.SGD,
    }
    if optimizer:
        assert optimizer.lower() in optimizers.keys(), "modulator optimizer=%s not supported" % optimizer
        optimizer = optimizers[optimizer.lower()]
        print("gradient_passing initialized with %s optimizer." % optimizer.__name__)
        optimizer = optimizer(A.mod.get_param_dicts() + A.demod.get_param_dicts())
    else:
        print("gradient_passing initialized WITHOUT an optimizer")
        optimizer = None

    batches_sent = 0
    results = []
    loss_criterion = torch.nn.CrossEntropyLoss()
    for i in range(num_iterations):
        preamble_labels = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[batch_size])
        preamble = integers_to_symbols(preamble_labels, bits_per_symbol)
        preamble = torch.from_numpy(preamble).float()
        ##MODULATE/action
        c_signal_forward = A.mod.modulate_tensor(preamble)
        ##CHANNEL
        N0 = get_N0(SNR_db=train_SNR_db, signal_power=signal_power)
        noise = torch.from_numpy(get_awgn(N0=N0, n=c_signal_forward.shape[0])).float()
        c_signal_forward_noisy = c_signal_forward + noise
        ##DEMODULATE/update and pass loss to mod
        preamble_halftrip_logits = A.demod.demodulate_tensor(c_signal_forward_noisy)
        loss = loss_criterion(input=preamble_halftrip_logits.float(), target=torch.from_numpy(preamble_labels))
        loss += A.mod.get_regularization_loss()
        loss += A.demod.get_regularization_loss()

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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
        'num_agents': 1,
    }
    return info, results
