import numpy as np

from utils.util_data import add_cartesian_awgn as add_awgn
from utils.util_data import get_grid_2d, get_test_SNR_dbs, integers_to_symbols


def get_random_preamble(n, bits_per_symbol):
    integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[n])
    return integers_to_symbols(integers, bits_per_symbol)


def get_ber(symbols1, symbols2):
    return np.sum(symbols1 ^ symbols2) / (symbols1.shape[0] * symbols1.shape[1])


def get_ser(symbols1, symbols2):
    diff = np.sum(symbols1 ^ symbols2, axis=1)
    diff[diff > 0] = 1
    return np.sum(diff) / len(symbols1)


def roundtrip_evaluate(*,
                       agent1,
                       agent2=None,
                       bits_per_symbol,
                       test_batch_size: int,
                       signal_power: float,
                       verbose: bool = False,
                       completed_iterations: int = None,
                       total_iterations: int = None,
                       **kwargs):
    grid_2d = get_grid_2d(grid=[-1.5, 1.5], points_per_dim=100)  # For getting demod boundaries
    test_SNR_dbs = get_test_SNR_dbs()[bits_per_symbol]['ber_roundtrip']
    A = agent1
    if agent2 is None:
        B = agent1
    else:
        B = agent2
    # Calculate Roundtrip Testing Accuracy on different SNRs
    preamble = get_random_preamble(n=test_batch_size, bits_per_symbol=bits_per_symbol)
    c_signal_forward = A.mod.modulate(preamble, mode='exploit', dtype='cartesian')
    _c_signal_forward = B.mod.modulate(preamble, mode='exploit', dtype='cartesian')

    test_bers = [[], []]
    test_sers = [[], []]
    for test_SNR_db in test_SNR_dbs:
        c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=test_SNR_db, signal_power=signal_power)
        preamble_halftrip = B.demod.demodulate(c_signal_forward_noisy)  #
        c_signal_backward = B.mod.modulate(preamble_halftrip, mode='exploit', dtype='cartesian')
        c_signal_backward_noisy = add_awgn(c_signal_backward, SNR_db=test_SNR_db, signal_power=signal_power)
        preamble_roundtrip = A.demod.demodulate(c_signal_backward_noisy)
        test_bers[0].append(float(get_ber(preamble, preamble_roundtrip)))
        test_sers[0].append(float(get_ser(preamble, preamble_roundtrip)))

        if not agent2 is None:
            _c_signal_forward_noisy = add_awgn(_c_signal_forward, SNR_db=test_SNR_db, signal_power=signal_power)
            _preamble_halftrip = A.demod.demodulate(_c_signal_forward_noisy)  #
            _c_signal_backward = A.mod.modulate(_preamble_halftrip, mode='exploit', dtype='cartesian')
            _c_signal_backward_noisy = add_awgn(_c_signal_backward, SNR_db=test_SNR_db, signal_power=signal_power)
            _preamble_roundtrip = B.demod.demodulate(_c_signal_backward_noisy)
            test_bers[1].append(float(get_ber(preamble, _preamble_roundtrip)))
            test_sers[1].append(float(get_ser(preamble, _preamble_roundtrip)))

    if agent2 is not None:
        avg_test_sers = np.mean([test_sers[0], test_sers[1]], axis=0)
        avg_test_bers = np.mean([test_bers[0], test_bers[1]], axis=0)
    else:
        avg_test_sers = test_sers[0]
        avg_test_bers = test_bers[0]
    if verbose is True:  # IF you want to manually debug:
        print(" ")
        if agent2 is not None:
            print("\t\t\t(%s --> %s --> %s), \n\t\t\t[%s --> %s --> %s], \n\t\t\t<Means>" % (
                A.name, B.name, A.name, B.name, A.name, B.name))
            for k in range(len(test_SNR_dbs)):
                print(
                    "Test SNR_db :% 5.1f | "
                    "(BER: %7.6f) [BER: %7.6f] <BER: %7.6f> | "
                    "(SER: %7.6f) [SER: %7.6f] <SER: %7.6f>"
                    % (test_SNR_dbs[k],
                       test_bers[0][k], test_bers[1][k], avg_test_bers[k],
                       test_sers[0][k], test_sers[1][k], avg_test_sers[k]))
        else:
            print("\t\t\t(%s --> %s --> %s)" % (A.name, B.name, A.name))
            for k in range(len(test_SNR_dbs)):
                print(
                    "Test SNR_db :% 5.1f | BER: %7.6f | SER: %7.6f" %
                      (test_SNR_dbs[k],
                       test_bers[0][k],
                       test_sers[0][k],))
        print(" ")
    elif (total_iterations is not None) and (completed_iterations is not None):
        print("[%s]" % ("." * completed_iterations + " " * (total_iterations - completed_iterations)), end='\r',
              flush=True)

    r2 = {}
    if agent2 is not None:
        r2 = {
            'test_bers_1': test_bers[0],
            'test_sers_1': test_sers[0],
            'test_bers_2': test_bers[1],
            'test_sers_2': test_sers[1],
            'mod_std_2': agent2.mod.get_std(),
            'constellation_2': agent2.mod.get_constellation(),
            'demod_grid_2': agent2.demod.get_demod_grid(grid_2d),
        }
    return {
        'test_SNR_dbs': test_SNR_dbs,
        'test_bers': avg_test_bers,  # mean
        'test_sers': avg_test_sers,  # mean
        'mod_std_1': agent1.mod.get_std(),
        'constellation_1': agent1.mod.get_constellation(),
        'demod_grid_1': agent1.demod.get_demod_grid(grid_2d),
        **r2
    }
