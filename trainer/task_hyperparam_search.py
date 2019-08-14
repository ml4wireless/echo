import argparse
import os
import sys

import hypertune
import numpy as np
import torch

ECHO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ECHO_DIR)
from models.agent import Agent
from utils.util_data import get_test_SNR_dbs, integers_to_symbols
from utils.util_data import add_cartesian_awgn as add_awgn
# from protocols.shared_preamble.trainer import trainer
from protocols.private_preamble.trainer import trainer

BER_TO_SNR = np.array([[1.00e-03, 1.04e+01],
                       [1.10e-02, 8.20e+00],
                       [2.10e-02, 7.40e+00],
                       [3.10e-02, 6.80e+00],
                       [4.10e-02, 6.20e+00],
                       [5.10e-02, 5.80e+00],
                       [6.10e-02, 5.40e+00],
                       [7.10e-02, 5.20e+00],
                       [8.10e-02, 4.80e+00],
                       [9.10e-02, 4.60e+00],
                       [1.01e-01, 4.20e+00],
                       [1.11e-01, 4.00e+00],
                       [1.21e-01, 3.80e+00],
                       [1.31e-01, 3.40e+00],
                       [1.41e-01, 3.20e+00],
                       [1.51e-01, 3.00e+00],
                       [1.61e-01, 2.80e+00],
                       [1.71e-01, 2.60e+00],
                       [1.81e-01, 2.20e+00],
                       [1.91e-01, 2.00e+00]])

def ber_to_snr(ber):
    if ber >= BER_TO_SNR[-1,0]:
        return 0
    bers = BER_TO_SNR[:, 0]
    index = np.argmin(np.abs(bers - ber))
    return bers[index][1]  # here is your result

BPS_TO_MOD_ORDER = {
    2: 'QPSK',
    3: '8PSK',
    4: 'QAM16'
}
TRAIN_SNRS_FOR_ORDER = {
    'QPSK': [13.0, 8.4, 4.2],
    '8PSK': [18.2, 13.2, 8.4],
    'QAM16': [20.0, 15.0, 10.4]
}
SNR_LEVEL_TO_INDEX = {
    'high': 0,
    'mid': 1,
    'low': 2
}


def get_ber(symbols1, symbols2):
    return np.sum(symbols1 ^ symbols2) / (symbols1.shape[0] * symbols1.shape[1])


def test(*,
         agent1,
         agent2,
         bits_per_symbol,
         test_SNR_db,
         signal_power=1.0,
         test_batch_size=5000,
         ):
    integers = np.random.randint(low=0, high=2 ** bits_per_symbol, size=[test_batch_size])
    preamble = integers_to_symbols(integers, bits_per_symbol=bits_per_symbol)
    A = agent1
    B = agent2
    c_signal_forward = A.mod.modulate(preamble, mode='exploit', dtype='cartesian')
    _c_signal_forward = B.mod.modulate(preamble, mode='exploit', dtype='cartesian')
    c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db=test_SNR_db, signal_power=signal_power)
    preamble_halftrip = B.demod.demodulate(c_signal_forward_noisy)  #
    c_signal_backward = B.mod.modulate(preamble_halftrip, mode='exploit', dtype='cartesian')
    c_signal_backward_noisy = add_awgn(c_signal_backward, SNR_db=test_SNR_db, signal_power=signal_power)
    preamble_roundtrip = A.demod.demodulate(c_signal_backward_noisy)

    _c_signal_forward_noisy = add_awgn(_c_signal_forward, SNR_db=test_SNR_db, signal_power=signal_power)
    _preamble_halftrip = A.demod.demodulate(_c_signal_forward_noisy)  #
    _c_signal_backward = A.mod.modulate(_preamble_halftrip, mode='exploit', dtype='cartesian')
    _c_signal_backward_noisy = add_awgn(_c_signal_backward, SNR_db=test_SNR_db, signal_power=signal_power)
    _preamble_roundtrip = B.demod.demodulate(_c_signal_backward_noisy)

    return (float(get_ber(preamble, preamble_roundtrip)) + float(get_ber(preamble, _preamble_roundtrip)))/2.0


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Poly Echo')
    parser = add_experiment_args(parser)
    parser = add_mod_args(parser)
    parser = add_demod_args(parser)
    parser = add_poly_demod_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cpu')

    agent_dict = {
        "bits_per_symbol": args.bits_per_symbol,
        "optimizer": "adam",
        "max_amplitude": 1.0,
        "demod_model": "poly",
        "mod_model": "poly",
        "mod_params": {
            "bits_per_symbol": args.bits_per_symbol,
            "stepsize_mu": args.stepsize_mu,
            "stepsize_sigma": args.stepsize_sigma,
            "initial_std": args.initial_std,
            "max_std": args.max_std,
            "min_std": args.min_std,
            "lambda_center": args.lambda_center,
            "lambda_l1": args.lambda_l1_mod,
            "lambda_l2": args.lambda_l2_mod,
            "lambda_baseline": args.lambda_baseline,
            "restrict_energy": 1,
            "max_amplitude": 1.0,
            "optimizer": "adam"
        },
        "demod_params": {
            "bits_per_symbol": args.bits_per_symbol,
            "optimizer": "adam",
            "stepsize_cross_entropy": args.stepsize_cross_entropy,
            "cross_entropy_weight": args.cross_entropy_weight,
            "epochs": args.epochs,
            "degree_polynomial": args.degree_polynomial,
            "lambda_l1": args.lambda_l1_demod,
            "lambda_l2": args.lambda_l2_demod,
        },
    }
    print(agent_dict)

    agent1 = Agent(agent_dict=agent_dict, name='Poly')
    agent2 = Agent(agent_dict=agent_dict, name='Clone')

    A = agent1
    B = agent2
    last = False
    total_batches_sent = 0
    test_SNR_dbs = get_test_SNR_dbs()[args.bits_per_symbol]['ber_roundtrip']
    test_SNR = test_SNR_dbs[4]
    batches_interval = 0
    while total_batches_sent <= args.total_batches:
        if batches_interval == 0 or batches_interval - args.log_interval >= 0 or total_batches_sent == args.total_batches:
            batches_interval = 0
            roundtrip_ber = test(
                agent1=agent1,
                agent2=agent2,
                bits_per_symbol=args.bits_per_symbol,
                test_SNR_db=test_SNR,
                signal_power=1.0,
            )
            print('Test: Roundtrip BER: {:.4f}'.format(roundtrip_ber))
            hpt = hypertune.HyperTune()
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='roundtrip_ber',
                metric_value=roundtrip_ber,
                global_step=args.batch_size * total_batches_sent
            )
            hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag='db_off',
                metric_value=test_SNR-ber_to_snr(roundtrip_ber),
                global_step=args.batch_size * total_batches_sent
            )
            # hpt.report_hyperparameter_tuning_metric(
            #     hyperparameter_metric_tag='agent1_centering_loss',
            #     metric_value=agent1.mod.model.location_loss(),
            #     global_step=args.batch_size * total_batches_sent
            # )
            # hpt.report_hyperparameter_tuning_metric(
            #     hyperparameter_metric_tag='agent2_centering_loss',
            #     metric_value=agent2.mod.model.location_loss(),
            #     global_step=args.batch_size * total_batches_sent
            # )
        B, A, batches_sent = trainer(agents=[A, B],
                                     bits_per_symbol=args.bits_per_symbol,
                                     batch_size=args.batch_size,
                                     train_SNR_db=TRAIN_SNRS_FOR_ORDER[BPS_TO_MOD_ORDER[args.bits_per_symbol]][
                                         SNR_LEVEL_TO_INDEX[args.train_snr]],
                                     signal_power=1.0,
                                     backwards_only=total_batches_sent == args.total_batches
                                     )
        total_batches_sent += batches_sent
        batches_interval += batches_sent


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
        '--lambda-baseline',
        type=float,
        default=0.0,
        metavar='L',
        help='Weight adding a baseline (default: 0.0)')
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
        default=2,
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
        default=600,
        metavar='N',
        help='how many batches to wait before logging training status (default: 250)')
    parser.add_argument(
        '--total-batches',
        type=int,
        default=20000,
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
