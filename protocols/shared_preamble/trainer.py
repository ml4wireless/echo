###SHARED PREAMBLE###
from utils.util_data import integers_to_symbols, add_cartesian_awgn as add_awgn
import numpy as np

###SHARED PREAMBLE###
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
        preamble_roundtrip = B.demod.demodulate(c_signal_backward_noisy)
        # Update mod after a roundtrip pass
        B.mod.update(B.preamble, B.actions, preamble_roundtrip)
        batches_sent += 1

    # guess of new preamble
    B.to_echo = B.demod.demodulate(c_signal_forward_noisy)
    # Update demod after a oneway pass
    if not backwards_only:
        B.demod.update(c_signal_forward_noisy, preamble)
        batches_sent += 1

    return A, B, batches_sent

    #
    #     ############### STATS ##########################
    #     if i % results_every == 0 or i == num_iterations:
    #         if verbose:
    #             print("ITER %i: Train SNR_db:% 5.1f" % (i, train_SNR_db))
    #
    #         result = evaluate(agent1=agents[0],
    #                           agent2=agents[1],
    #                           bits_per_symbol=bits_per_symbol,
    #                           signal_power=signal_power,
    #                           verbose=verbose or i == num_iterations,
    #                           total_iterations=num_iterations // results_every,
    #                           completed_iterations=i//results_every,
    #                           **kwargs)
    #
    #         test_SNR_dbs = result['test_SNR_dbs']
    #         test_bers = result['test_bers']
    #         db_off_for_test_snr = [testSNR - br.get_optimal_SNR_for_BER_roundtrip(testBER, bits_per_symbol)
    #                                for testSNR, testBER in zip(test_SNR_dbs, test_bers)]
    #         ###ADD TO RESULT
    #         result['batches_sent'] = batches_sent
    #         result['db_off'] = db_off_for_test_snr
    #         results += [result]
    #         if early_stopping and all(np.array(db_off_for_test_snr) <= early_stopping_db_off):
    #             print("STOPPED AT ITERATION: %i" % i)
    #             print(['0 BER', '1e-5 BER', '1e-4 BER', '1e-3 BER', '1e-2 BER', '1e-1 BER'])
    #             print("TEST SNR dBs : ", test_SNR_dbs)
    #             print("dB off Optimal : ", db_off_for_test_snr)
    #             print("Early Stopping dBs off: %d" % early_stopping_db_off)
    #             early_stop = True
    #             break
    # info = {
    #     'bits_per_symbol': bits_per_symbol,
    #     'train_SNR_db': train_SNR_db,
    #     'num_results': len(results),
    #     'test_SNR_dbs': test_SNR_dbs,
    #     'early_stop': early_stop,
    #     'early_stop_threshold_db_off': early_stopping_db_off,
    #     'batch_size': batch_size,
    #     'num_agents': 2,
    # }