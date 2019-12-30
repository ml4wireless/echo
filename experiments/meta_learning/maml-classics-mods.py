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

from models.modulator_models.MAMLNeural import MAMLNeural
# from models.maml_modulator import MAMLModulatorGradientPassing
from models.maml_modulator import MAMLModulatorLossPassing
from utils.util_data import integers_to_symbols
from utils.util_data import torch_tensor_to_numpy as to_numpy
from utils.util_data import numpy_to_torch_tensor as to_tensor
from utils.util_meta import ClassicDemodDatasetThreaded
from utils.util_pandas import PandasDF, create_agent_from_row


def visualize_constellation(data_c, labels, legends=True):
    for label in np.unique(labels):
        cur_data_c = data_c[labels == label, :]
        plt.scatter(cur_data_c[:, 0], cur_data_c[:, 1], label=label)
    if legends:
        plt.legend()
    plt.show()


def plot_constellation(mod, params=None):
    data_si = np.arange(2**mod.bits_per_symbol)
    data_sb = to_tensor(integers_to_symbols(data_si, mod.bits_per_symbol)).float()
    data_c = mod.model.forward(data_sb, params)
    data_c = to_numpy(data_c)

    for label in np.unique(data_si):
        cur_data_c = data_c[data_si == label, :]
        plt.scatter(cur_data_c[:, 0], cur_data_c[:, 1], label=label)
        plt.annotate(label, (cur_data_c[:, 0], cur_data_c[:, 1]))
    plt.show()


############### TRAIN ################
def train(mod_args, bits_per_symbol=2, ntasks=16, epochs=500, reduce_std_every=25, reduce_std_by=0.9, disable_plots=False):
    dataset = ClassicDemodDatasetThreaded(bits_per_symbol=bits_per_symbol, ntasks=ntasks)
    mm = MAMLModulatorLossPassing(**mod_args)
    params_rand = mm.model.base.state_dict().copy()
    stats_every = 100
    # reduce_std_every = 25
    # reduce_std_by = 0.88
    losses = []
    bers = []
    with tqdm(enumerate(dataset), desc="Epoch", total=epochs) as pbar:
        for e, demods in pbar:
            if e > 0 and e % reduce_std_every == 0:
                mm.std_explore *= reduce_std_by
            if e % stats_every == 0 or e == epochs - 1:
                print("Epoch", e)
                mm.verbose = True
                loss, ber = mm.update_maml(demods)
                # print(loss, ber)
                # plot_constellation(mm)
            else:
                mm.verbose = False
                loss, ber = mm.update_maml(demods)
            losses.append(loss)
            bers.append(ber)
            postfix = OrderedDict()
            postfix['Loss'] = loss
            postfix['BER'] = ber
            pbar.set_postfix(postfix)
            if e >= epochs:
                break

    params_maml = mm.model.base.state_dict().copy()

    if not disable_plots:
        torch.save(params_maml, "./plots/pg-maml-params-bps{}.mdl".format(bits_per_symbol))
        print("Plotting train results...")
        plt.plot(losses, 'o-')
        plt.savefig("./plots/mod_train_losses_maml.png")
        plt.close()

        plt.plot(bers, 'o-')
        plt.yscale('log')
        plt.savefig("./plots/mod_train_bers_maml.png")
        plt.close()
        print("Finished plotting train results")
    return mm, params_rand, params_maml, losses, bers


############### TEST ################
def test_nshots(mod, params, nshots, init="MAML", bits_per_symbol=2, ntasks=16, disable_plots=False):
    dataset = ClassicDemodDatasetThreaded(bits_per_symbol=bits_per_symbol, ntasks=ntasks)
    demods = dataset[None]
    test_bers = []
    print("Test std_explore={}".format(mod.std_explore))
    with tqdm(enumerate(demods), desc="Test Task", total=ntasks) as pbar:
        for t, demod in pbar:
            mod.model.base.load_state_dict(params)
            taskbers = []
            taskbers = mod.update_test(demod, nshots=nshots, step_size=mod.stepsize_inner, batch_size=mod.inner_batch_size)
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
        plt.savefig("./plots/mod_bers_{}shots_{}.png".format(nshots, init.lower()))
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
    p.add_argument("-ns", "--nshots", dest='nshots', type=int, default=200, help="Number of shots when testing (default: %(default)s)")
    p.add_argument("--snr", dest='SNR_db', type=float, default=15., help="Training SNR dB (default: %(default)s)")
    p.add_argument("-sm", "--stepsize-meta", dest='stepsize_meta', type=float, default=1e-1, help="Step size for meta updates (default: %(default)s)")
    p.add_argument("-si", "--stepsize-inner", dest='stepsize_inner', type=float, default=1e-1, help="Step size for inner updates (default: %(default)s)")
    p.add_argument("--outer-batch-size", type=int, default=1024, help="Number for symbols used for outer update (default: %(default)s)")
    p.add_argument("--inner-batch-size", type=int, default=1024, help="Number for symbols used for inner update (default: %(default)s)")
    p.add_argument("--inner-steps", type=int, default=10, help="Number of inner steps to take before outer updates (default: %(default)s)")
    p.add_argument("--first-order", action="store_true", help="First order approximation to MAML")
    p.add_argument("--restrict-energy", type=int, default=1, help="Energy restriction method")
    p.add_argument("--optimizer", default='sgd', choices=['sgd', 'adam'], help="Name of optimizer for meta update (default: %(default)s)")
    p.add_argument("-std", "--standard-explore", dest='std_explore', type=float, default=3e-1, help="Std dev for exploration (default: %(default)s)")
    p.add_argument("--reduce-std-every", type=int, default=50, help="Reduce std_explore every N iterations, typically chosen so that std_explore=0.3 after all epochs (default: %(default)s)")
    p.add_argument("--reduce-std-by", type=float, default=1.0, help="Reduce std_explore multiplicatively, typically chosen so that std_explore=0.3 after all epochs (default: %(default)s)")
    p.add_argument("-l", "--hidden-layers", dest='hidden_layers', nargs="+", type=int, default=[100, 100], help="List of hidden layer sizes (default: %(default)s)")
    p.add_argument("--loss", dest="loss_function", choices=["vanilla_pg", "ppo"], default="vanilla_pg", help="Loss function for Policy Gradient updates (default: %(default)s)")
    p.add_argument("--ppo-clip-ratio", type=float, default=0.2, help="PPO ratio clipping parameter (default: %(default)s)")
    p.add_argument("--ppo-epochs", type=int, default=5, help="PPO updates per batch (default: %(default)s)")
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
    mod_args = vars(args)
    mod_args['model'] = MAMLNeural
    # Pop disable_plots because we don't want it as a search key
    disable_plots = mod_args.pop("disable_plots")
    reduce_std_every = mod_args["reduce_std_every"]
    reduce_std_by = mod_args["reduce_std_by"]

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

    params_dict = mod_args.copy()
    params_dict.pop("device")  # We don't care about device for results
    params_dict.pop("verbose")  # We don't care about verbosity for results
    params_dict['model'] = params_dict['model'].__name__
    params_dict['learner'] = "modulator"  # Or demodulator
    params_dict['meta-alg'] = "MAML"  # Or REPTILE
    response = DF.search_rows(params_dict)
    if response is None or len(response) == 0:
        print("model not found")
        print("Training with MAML for {} epochs".format(epochs))
        mod, rand_params, maml_params, losses, train_bers = train(
                mod_args, bps, ntasks, epochs=epochs, 
                reduce_std_every=reduce_std_every, 
                reduce_std_by=reduce_std_by, 
                disable_plots=disable_plots)

        print("Adapting from MAML initialization for {} shots".format(nshots))
        maml_bers = test_nshots(mod, maml_params, nshots, "MAML", bps, ntasks, disable_plots=disable_plots)
        print("Final MAML BER {}".format(np.mean(maml_bers[-1, :])))
        print("Adapting from RAND initialization for {} shots".format(nshots))
        rand_bers = test_nshots(mod, rand_params, nshots, "RAND", bps, ntasks, disable_plots=disable_plots)
        print("Final RAND BER {}".format(np.mean(rand_bers[-1, :])))

        results_dict = {'maml-test-bers': maml_bers, 'rand-test-bers': rand_bers}
        results_dict['train-losses'] = losses
        results_dict['train-bers'] = train_bers
        # Pass model and results into add_row call to get data race safety
        model_filename, results_filename = DF.add_row(params_dict, model=mod.get_weights(), results=results_dict)
    else:
        print("model found")
        filename = response.iloc[0]['results_filename']
#         print("Retrieving results from file, ", filename)
        results_dict = np.load(filename, allow_pickle=True).item()
        mod = create_agent_from_row(response.iloc[0], MAMLModulatorLossPassing, MAMLNeural, {'device': mod_args['device']})


if __name__ == "__main__":
    main()
    sys.exit(0)
