#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import torch
from collections import OrderedDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

sys.path.append(os.getcwd())

from models.demodulator_models.MAMLNeural import MAMLNeural
# from models.maml_modulator import MAMLModulatorGradientPassing
from utils.util_data import integers_to_symbols
from utils.util_data import torch_tensor_to_numpy as to_numpy
from utils.util_data import numpy_to_torch_tensor as to_tensor
from utils.util_meta import ClassicModDatasetThreaded
from models.maml_demodulator import MAMLDemodulator
from utils.util_data import integers_to_symbols, get_grid_2d
from utils.util_pandas import PandasDF, create_agent_from_row


def plot_boundary(demod, bps, lims=[-1.5, 1.5]):
    grid = get_grid_2d(lims)
    classes = demod.get_demod_grid(grid)
    nlabels = int(2 ** bps)
    colors = plt.rcParams['axes.prop_cycle'][:nlabels]
    colors = [c['color'] for c in colors]
    while len(colors) < nlabels:
        colors *= 2
    colors = np.array(colors)
    plt.scatter(grid[:, 0], grid[:, 1], c=colors[classes])
    for c in np.unique(classes):
        c_grid = grid[classes == c, :]
        center = np.mean(c_grid, axis=0)
        plt.annotate(c, center, fontsize=12, fontweight='bold')
    plt.title("Demod Grid")
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.show()


############### TRAIN ################
def train(demod_args, bits_per_symbol=2, ntasks=16, epochs=500, disable_plots=False):
    dataset = ClassicModDatasetThreaded(bits_per_symbol=bits_per_symbol, ntasks=ntasks)
    md = MAMLDemodulator(**demod_args)
    params_rand = md.model.base.state_dict().copy()
    stats_every = 100
    losses = []
    bers = []
    with tqdm(enumerate(dataset), desc="Epoch", total=epochs) as pbar:
        for e, demods in pbar:
            if e % stats_every == 0 or e == epochs - 1:
                print("Epoch", e)
                md.verbose = True
                loss, ber = md.update_maml(demods)
                # print(loss, ber)
                # plot_constellation(md)
            else:
                md.verbose = False
                loss, ber = md.update_maml(demods)
            losses.append(loss)
            bers.append(ber)
            postfix = OrderedDict()
            postfix['Loss'] = loss
            postfix['BER'] = ber
            pbar.set_postfix(postfix)
            if e >= epochs:
                break

    params_maml = md.model.base.state_dict().copy()

    if not disable_plots:
        torch.save(params_maml, "./plots/demod-maml-params-bps{}.mdl".format(bits_per_symbol))
        print("Plotting train results...")
        plt.plot(losses, 'o-')
        plt.savefig("./plots/demod_train_losses_maml.png")
        plt.close()

        plt.plot(bers, 'o-')
        plt.yscale('log')
        plt.savefig("./plots/demod_train_bers_maml.png")
        plt.close()
        print("Finished plotting train results")
    return md, params_rand, params_maml, losses, bers


############### TEST ################
def test_nshots(demod, params, nshots, init="MAML", bits_per_symbol=2, ntasks=16, disable_plots=False):
    dataset = ClassicModDatasetThreaded(bits_per_symbol=bits_per_symbol, ntasks=ntasks)
    mods = dataset[None]

    test_bers = []
    with tqdm(enumerate(mods), desc="Test Task", total=ntasks) as pbar:
        for t, mod in pbar:
            demod.model.base.load_state_dict(params)
            taskbers = []
            taskbers = demod.update_test(mod, nshots=nshots, step_size=demod.stepsize_inner, batch_size=demod.inner_batch_size)
            test_bers.append(taskbers)
    # print(test_bers)
    bers = np.array(test_bers).T

    if not disable_plots:
        print("Plotting test results...")
        plt.plot(np.arange(nshots + 1), bers, alpha=0.4)
        plt.plot(np.arange(nshots + 1), np.mean(bers, axis=1), 'k:', linewidth=3, label="Mean BER Across Tasks")
        plt.yscale('log')
        plt.legend()
        plt.title("{} Initialization".format(init))
        # _ = plt.xticks(np.arange(10) * 5)
        plt.savefig("./plots/demod_bers_{}shots_{}.png".format(nshots, init.lower()))
        plt.close()
        print("Finished plotting test results")
    return bers


############### MAIN ################
def parse_args():
    p = argparse.ArgumentParser(description="MAML pretraining and testing")
    p.add_argument("-e", "--epochs", dest='epochs', type=int, default=500, help="Number of training epochs (default: %(default)s)")
    p.add_argument("-v", "--verbose", dest='verbose', action="store_true", help="Print more verbose messages")
    p.add_argument("-bps", "--bits-per-symbol", dest='bits_per_symbol', type=int, default=2, help="Number of bits per symbol (default: %(default)s)")
    p.add_argument("-t", "--ntasks", dest='ntasks', type=int, default=16, help="Number of demod tasks per meta update (default: %(default)s)")
    p.add_argument("-ns", "--nshots", dest='nshots', type=int, default=20, help="Number of shots when testing (default: %(default)s)")
    p.add_argument("--snr", dest='SNR_db', type=float, default=15., help="Training SNR dB (default: %(default)s)")
    p.add_argument("-sm", "--stepsize-meta", dest='stepsize_meta', type=float, default=1e-1, help="Step size for meta updates (default: %(default)s)")
    p.add_argument("-si", "--stepsize-inner", dest='stepsize_inner', type=float, default=1e-1, help="Step size for inner updates (default: %(default)s)")
    p.add_argument("--outer-batch-size", type=int, default=1024, help="Number for symbols used for outer update (default: %(default)s)")
    p.add_argument("--inner-batch-size", type=int, default=1024, help="Number for symbols used for inner update (default: %(default)s)")
    p.add_argument("--inner-steps", type=int, default=10, help="Number of inner steps to take before outer updates (default: %(default)s)")
    p.add_argument("--first-order", action="store_true", help="First order approximation to MAML")
    p.add_argument("--optimizer", default='sgd', choices=['sgd', 'adam'], help="Name of optimizer for meta update (default: %(default)s)")
    p.add_argument("-l", "--hidden-layers", dest='hidden_layers', nargs="+", type=int, default=[100, 100], help="List of hidden layer sizes (default: %(default)s)")
    p.add_argument("-d", "--device", dest='device', default='cpu', help="PyTorch device flag")
    p.add_argument("--disable-plots", dest='disable_plots', action="store_true", help="Disable plotting (useful on headless machines where plotting is slow)")
    p.add_argument("--seed", type=int, default=777, help="RNG seed")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    bps = args.bits_per_symbol
    nshots = args.nshots
    ntasks = args.ntasks
    epochs = args.epochs
    demod_args = vars(args)
    demod_args['model'] = MAMLNeural
    # Pop disable_plots because we don't want it as a search key
    disable_plots = demod_args.pop("disable_plots")

    # Save run results
    try:
        os.mkdir("./plots")
    except FileExistsError:
        pass
    HOME_DIR = './data'
    PANDAS_DATAFRAME_FILENAME = HOME_DIR + '/df.xls'
    RESULTS_DIR = HOME_DIR + '/results'
    RESULTS_KEY = 'results_filename'
    MODELS_DIR = HOME_DIR + '/saved_models'
    MODELS_KEY = 'model_filename'

    DF = PandasDF(HOME_DIR=HOME_DIR, RESULTS_DIR=RESULTS_DIR, MODELS_DIR=MODELS_DIR,
                  PANDAS_DATAFRAME_FILENAME=PANDAS_DATAFRAME_FILENAME,
                  RESULTS_KEY=RESULTS_KEY, MODELS_KEY=MODELS_KEY)

    params_dict = demod_args.copy()
    params_dict.pop("device")  # We don't care about device for results
    params_dict.pop("verbose")  # We don't care about verbosity for results
    params_dict['model'] = params_dict['model'].__name__
    params_dict['learner'] = "demodulator"  # Or demodulator
    params_dict['meta-alg'] = "MAML"  # Or REPTILE?
    response = DF.search_rows(params_dict)
    if response is None or len(response) == 0:
        print("model not found")
        print("Training with MAML for {} epochs".format(epochs))
        demod, rand_params, maml_params, losses, train_bers = train(demod_args, bps, ntasks, epochs=epochs, disable_plots=disable_plots)

        print("Adapting from MAML initialization for {} shots".format(nshots))
        maml_bers = test_nshots(demod, maml_params, nshots, "MAML", bps, ntasks, disable_plots=disable_plots)
        print("Final MAML BER {}".format(np.mean(maml_bers[-1, :])))
        print("Adapting from RAND initialization for {} shots".format(nshots))
        rand_bers = test_nshots(demod, rand_params, nshots, "RAND", bps, ntasks, disable_plots=disable_plots)
        print("Final RAND BER {}".format(np.mean(rand_bers[-1, :])))

        results_dict = {'maml-test-bers': maml_bers, 'rand-test-bers': rand_bers}
        results_dict['train-losses'] = losses
        results_dict['train-bers'] = train_bers
        # Pass model and results into add_row call to get data race safety
        model_filename, results_filename = DF.add_row(params_dict, model=demod.get_weights(), results=results_dict)
    else:
        print("model found")
        filename = response.iloc[0]['results_filename']
#         print("Retrieving results from file, ", filename)
        results_dict = np.load(filename, allow_pickle=True).item()
        demod = create_agent_from_row(response.iloc[0], MAMLDemodulator, MAMLNeural, {'device': demod_args['device']})


if __name__ == "__main__":
    main()
    sys.exit(0)
