#This was abandoned in favor of using the genomics pipeline which is cheaper and faster to spin up.
#The only problem is that the genomics compute sometimes stalls and you have to do a manual stop and start up the missing jobs


import argparse
import os
import sys, json
# import hypertune
from importlib import import_module
import numpy as np
import torch

ECHO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
WORK_DIR = os.path.join(ECHO_DIR, "trainer", "work")
sys.path.append(ECHO_DIR)
from models.agent import Agent
from utils.util_data import get_test_SNR_dbs, integers_to_symbols
from utils.util_data import add_cartesian_awgn as add_awgn


def get_ber(symbols1, symbols2):
    return np.sum(symbols1 ^ symbols2) / (symbols1.shape[0] * symbols1.shape[1])


def test(*,
         agent1,
         agent2,
         bits_per_symbol,
         test_SNR_dbs,
         signal_power=1.0,
         test_batch_size=10000,
         ):
    integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[test_batch_size])
    preamble = integers_to_symbols(integers, bits_per_symbol=bits_per_symbol)
    A = agent1
    B = agent2 if agent2 is not None else agent1
    c_signal_forward = A.mod.modulate(preamble, mode='exploit', dtype='cartesian')
    _c_signal_forward = B.mod.modulate(preamble, mode='exploit', dtype='cartesian')
    for test_SNR_db in test_SNR_dbs:
        c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=test_SNR_db, signal_power=signal_power)
        preamble_halftrip = B.demod.demodulate(c_signal_forward_noisy)  #
        c_signal_backward = B.mod.modulate(preamble_halftrip, mode='exploit', dtype='cartesian')
        c_signal_backward_noisy = add_awgn(c_signal_backward, SNR_db=test_SNR_db, signal_power=signal_power)
        preamble_roundtrip = A.demod.demodulate(c_signal_backward_noisy)
        if agent2 is not None:
            _c_signal_forward_noisy = add_awgn(_c_signal_forward, SNR_db=test_SNR_db, signal_power=signal_power)
            _preamble_halftrip = A.demod.demodulate(_c_signal_forward_noisy)  #
            _c_signal_backward = A.mod.modulate(_preamble_halftrip, mode='exploit', dtype='cartesian')
            _c_signal_backward_noisy = add_awgn(_c_signal_backward, SNR_db=test_SNR_db, signal_power=signal_power)
            _preamble_roundtrip = B.demod.demodulate(_c_signal_backward_noisy)
    return "TODOTODOTODO"
    # return float(get_ber(preamble, preamble_roundtrip)) + float(get_ber(preamble, _preamble_roundtrip))

def prepare_environment(params):
    seed = params.pop("random_seed", 13370)
    numpy_seed = params.pop("numpy_seed", 1337)
    torch_seed = params.pop("pytorch_seed", 133)
    if seed is not None:
        import random
        random.seed(seed)
    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    torch.set_num_threads(1)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Poly Echo')
    parser = add_experiment_args(parser)
    args = parser.parse_args()
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    params_file = os.path.join(WORK_DIR, '%s.json'%args.task_id)
    with open(params_file) as file:
        params = json.load(file)

    keys = params.keys()
    agent_keys = [key for key in keys if 'agent' in key]
    num_agents = len(agent_keys)
    meta = params.pop('__meta__')
    verbose = meta['verbose']
    trial_num = meta['trial_num']
    protocol = meta['protocol']
    experiment_name = meta['experiment_name']
    prepare_environment(meta)
    agents = []
    for agent_key in agent_keys:
        agent_params = params.pop(agent_key)
        agents += [Agent(agent_dict=agent_params, name=agent_key)]

    if num_agents == 2:
        A = agents[0]
        B = agents[1]
    else:
        A = agents[0]
        B = agents[0]
    # Load Protocol
    module_name = 'protocols.%s.trainer' % (protocol)
    trainer = getattr(import_module(module_name), 'trainer')
    total_batches_sent = 0
    results, symbols_sent = [], []
    test_SNR_dbs = get_test_SNR_dbs()[params['bits_per_symbol']]['ber_roundtrip']
    num_iterations = params['num_iterations']
    total_iterations = num_iterations + (1 if num_agents == 2 else 0)
    for i in range(total_iterations):
        B, A, batches_sent = trainer(agents=[A, B],
                                     bits_per_symbol=params['bits_per_symbol'],
                                     batch_size=params['batch_size'],
                                     train_SNR_db=params['train_SNR_db'],
                                     signal_power=params['signal_power'],
                                     backwards_only=(i == num_iterations)
                                     )
        total_batches_sent += batches_sent
        if i % params['results_every'] == 0:
            result = test(
                agent1=agents[0],
                agent2=agents[1] if num_agents == 2 else None,
                bits_per_symbol=params['bits_per_symbol'],
                test_SNR_dbs=test_SNR_dbs,
                signal_power=params['signal_power'],
                test_batch_size=params['test_batch_size']
            )
            test_bers = result['test_bers']
            print('Test: Roundtrip BER: ', test_bers)
        symbols_sent += [total_batches_sent * params['batch_size']]
    results += [result]


def add_mod_args(parser):
    """Adds arguments for modulator
    """
    parser.add_argument(
        '--stepsize-mu',
        type=float,
        default=0.001,
        metavar='LR',
        help='learning rate for mean of gaussian policy (default: 0.001)')
    parser.add_argument(
        '--stepsize-sigma',
        type=float,
        default=0.001,
        metavar='LR',
        help='learning rate for std of gaussian policy (default: 0.001)')
    parser.add_argument(
        '--initial-std',
        type=float,
        default=0.5,
        metavar='STD',
        help='std for gaussian policy (default: 0.5)')
    parser.add_argument(
        '--min-std',
        type=float,
        default=0.0001,
        metavar='STD',
        help='Min std for gaussian policy (default: 0.0001)')
    parser.add_argument(
        '--max-std',
        type=float,
        default=100.0,
        metavar='STD',
        help='Max std for gaussian policy (default: 100)')
    parser.add_argument(
        '--lambda-center',
        type=float,
        default=0.0,
        metavar='L',
        help='Weight penalizing un-centered constellations (default: 0.0)')
    parser.add_argument(
        '--lambda-l1-mod',
        type=float,
        default=0.0,
        metavar='L',
        help='L1 regularization weight (default: 0.0)')
    parser.add_argument(
        '--lambda-l2-mod',
        type=float,
        default=0.0,
        metavar='L',
        help='L2 regularization weight (default: 0.0)')
    return parser


def add_demod_args(parser):
    """Adds arguments for polynomial modulator
    """
    parser.add_argument(
        '--stepsize-cross-entropy',
        type=float,
        default=0.01,
        metavar='LR',
        help='learning rate for supervised demod learning (default: 0.001)')
    parser.add_argument(
        '--cross-entropy-weight',
        type=float,
        default=1.0,
        metavar='W',
        help='weight applied to update (default: 1.0)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        metavar='N',
        help='Num epochs (default: 5)')
    parser.add_argument(
        '--lambda-l1-demod',
        type=float,
        default=0.0,
        metavar='L',
        help='L1 regularization weight (default: 0.0)')
    parser.add_argument(
        '--lambda-l2-demod',
        type=float,
        default=0.0,
        metavar='L',
        help='L2 regularization weight (default: 0.0)')
    return parser


def add_poly_demod_args(parser):
    parser.add_argument(
        '--degree-polynomial',
        type=int,
        default=3,
        metavar='N',
        help='degree of polynomial (default: 3)')
    return parser


def add_experiment_args(parser):
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--train-snr',
        type=str,
        default='mid',
        metavar='LEVEL',
        choices=['high', 'mid', 'low'],
        help='signal to noise ratio for training (default: "mid")')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=250,
        metavar='N',
        help='how many batches to wait before logging training status (default: 250)')
    parser.add_argument(
        '--total-batches',
        type=int,
        default=10000,
        metavar='N',
        help='how many batches to train for (default: 10000)')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        metavar='N',
        help='How many symbols in a preamble (default: 32)')
    parser.add_argument(
        '--bits-per-symbol',
        type=int,
        default=2,
        metavar='N',
        choices=[2, 3, 4],
        help='Modulation scheme (default: 2 for QPSK)')
    parser.add_argument(
        '--model-dir',
        default=None,
        help='The directory to store the model')
    parser.add_argument('--job-dir',  # handled automatically by AI Platform
                        help='GCS location to write checkpoints and export ' \
                             'models')
    return parser


if __name__ == '__main__':
    main()
