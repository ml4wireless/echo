#!/usr/bin/env python

import sys
import subprocess as sp
from itertools import product
from concurrent.futures import ThreadPoolExecutor
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = 'ppo'
epochs = 250
ntasks = 16
nshots = 10
stepsize_metas = [1e-2, 3e-2, 1e-1]
stepsize_inners = [1e-2, 3e-2, 1e-1]
inner_stepss = [5, 10]
inner_batch_sizes = [256]
outer_batch_sizes = [1024]
first_orders = [True, False]
std_explores = [3e-1, 1]


def do_run(args):
    print("Starting run with args {}".format(args))
    print()
    print(" ".join(args) + ":")
    rv = sp.run(args, stdout=sys.stdout, stderr=sys.stderr)
    print()
    if rv.returncode != 0:
        print("Run exited with return code", rv.returncode)
    else:
        print("Finished run")


def gen_demod_runs(bps, snrs, hls):
    with ThreadPoolExecutor(max_workers=8) as tpe:
        for sm, si, ins, ibs, obs, fo, snr in product(
                stepsize_metas, stepsize_inners,
                inner_stepss,
                inner_batch_sizes, outer_batch_sizes,
                first_orders,
                snrs):
            args = ["python", "./experiments/meta_learning/maml-classics-demods.py",
                    "--disable-plots", "--device", device,
                    "-bps", str(bps), "--snr", str(snr), "--ntasks", str(ntasks),
                    "--nshots", str(nshots), "--epochs", str(epochs),
                    "--inner-steps", str(ins),
                    "--stepsize-meta", str(sm),
                    "--stepsize-inner", str(si),
                    "--inner-batch-size", str(ibs),
                    "--outer-batch-size", str(obs),
                    "--hidden-layers"]
            for hl in hls:
                args.append(str(hl))
            if fo:
                args.append("--first-order")

            tpe.submit(do_run, args)


def gen_mod_runs(bps, snrs, hls):
    with ThreadPoolExecutor(max_workers=4) as tpe:
        for sm, si, ins, ibs, obs, fo, sx, snr in product(
                stepsize_metas, stepsize_inners,
                inner_stepss,
                inner_batch_sizes, outer_batch_sizes,
                first_orders, std_explores,
                snrs):
            args = ["python", "./experiments/meta_learning/maml-classics-mods.py",
                    "--disable-plots", "--device", device,
                    "-bps", str(bps), "--snr", str(snr), "--ntasks", str(ntasks),
                    "--nshots", str(nshots), "--epochs", str(epochs),
                    "--loss", loss_fn,
                    "--stepsize-meta", str(sm),
                    "--stepsize-inner", str(si),
                    "--inner-steps", str(ins),
                    "--inner-batch-size", str(ibs),
                    "--outer-batch-size", str(obs),
                    "--standard-explore", str(sx),
                    "--hidden-layers"]
            for hl in hls:
                args.append(str(hl))
            if fo:
                args.append("--first-order")

            tpe.submit(do_run, args)

# No PPO loss function for demodulators
# ### Demodulators
# #########
# BPS = 6
# SNR_dbs = [20.8]
# hidden_layers = [128, 128]
# gen_demod_runs(BPS, SNR_dbs, hidden_layers)
# 
# #########
# BPS = 2
# SNR_dbs = [8.4]
# hidden_layers = [50, 50]
# gen_demod_runs(BPS, SNR_dbs, hidden_layers)
# 
# #########
# BPS = 4
# SNR_dbs = [15]
# hidden_layers = [100, 100]
# gen_demod_runs(BPS, SNR_dbs, hidden_layers)

### Modulators
#########
BPS = 2
SNR_dbs = [8.4]
hidden_layers = [50, 50]
gen_mod_runs(BPS, SNR_dbs, hidden_layers)

#########
BPS = 4
SNR_dbs = [15]
hidden_layers = [100, 100]
gen_mod_runs(BPS, SNR_dbs, hidden_layers)

#########
BPS = 6
SNR_dbs = [20.8]
hidden_layers = [128, 128]
gen_mod_runs(BPS, SNR_dbs, hidden_layers)

