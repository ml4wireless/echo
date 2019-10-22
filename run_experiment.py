import json
import os
import sys
import random
from copy import deepcopy
from importlib import import_module

import numpy as np
import torch

from models.agent import Agent

#FOR PROFILING SPEED OF CODE
# import cProfile as profile
# pr = profile.Profile()
# pr.disable()

## TODO
# 1. check that make_jobs in a batch  == make_jobs single (seeds are set properly)
# DONE 2. make sure that rerunning the same job is reproduible
# 3. make sure that rerunning the same job but with other stuff in between is reproducible
# DONE 4. check that results here == results in torch echo

ECHO_DIR = os.path.dirname(os.path.realpath(__file__))

def rm_mkdir(dir):
    if os.path.isdir(dir):
        import shutil
        shutil.rmtree(dir)
    os.makedirs(dir)
    return

def prepare_environment(params):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Pytorch.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reasonably reproducible. If you are using this from your own
    project, you will want to call this function before importing Pytorch. Complete determinism
    is very difficult to achieve with libraries doing optimized linear algebra due to massively
    parallel execution, which is exacerbated by using GPUs.
    Parameters
    ----------
    params: Params object or dict, required.
        A ``Params`` object or dict holding the json parameters.
    """
    seed = params.pop("random_seed", 13370)
    numpy_seed = params.pop("numpy_seed", 1337)
    torch_seed = params.pop("pytorch_seed", 133)

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
        # Seed all GPUs with the same seed if available.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
    device = torch.device("cpu")
    torch.set_num_threads(1)

    sys.path.append(ECHO_DIR)

# animated_plot()


def run(jobs_file, job_id=None, plot=False, echo_symlink_to=None, job_date=None):
    # pr.enable()
    with open(jobs_file) as jfile:
        jobs = json.load(jfile)
    if isinstance(jobs, dict):
        # ToDo: make this more explicit, but basically, you can give me a json of SINGLE param dict, aka rerun a
        #       single job that was spit out (ex: echo/experiments/gradient_passing/QPSK_neural_and_neural/results/0.json)
        jobs = [jobs]
    elif job_id is not None:  # 0 = False you dummy
        plot = plot
        jobs = [jobs[job_id]]
    else:
        # NO PLOTTING IF YOU ARE RUNNING A BUNCH OF JOBS...NO!
        plot = False
    for params in jobs:
        params_copy = deepcopy(params)
        keys = params.keys()
        agent_keys = [key for key in keys if 'agent' in key]
        meta = params.pop('__meta__')
        verbose = meta['verbose']
        job_id = meta['job_id']
        trial_num = meta['trial_num']
        protocol = meta['protocol']
        experiment_name = meta['experiment_name']
        experiment_dir = os.path.abspath(os.path.join(ECHO_DIR, 'experiments', protocol, experiment_name))
        results_dir = os.path.abspath(os.path.join(experiment_dir, 'results'))
        # DEAL WITH SYMLINKING FOR RUNNING ON BRC
        if echo_symlink_to is not None:
            assert os.path.isdir(echo_symlink_to), "Invalid symlink path"
            if os.path.isdir(results_dir):
                old_results_dir = os.path.abspath(os.path.join(experiment_dir, 'old_results'))
                os.makedirs(old_results_dir, exist_ok=True)
                n = len(os.listdir(old_results_dir))
                os.rename(results_dir, os.path.abspath(os.path.join(old_results_dir, '%i' % n)))
            _experiment_dir = os.path.abspath(
                os.path.join(echo_symlink_to, 'experiments', protocol, experiment_name))
            if job_date is None:
                job_date = "results"
            _results_dir = os.path.abspath(os.path.join(_experiment_dir, job_date))
            os.makedirs(_results_dir, exist_ok=True)
            if not (os.path.islink(results_dir)
                    and os.readlink(results_dir) == _results_dir):
                os.symlink(_results_dir, results_dir)
        else:
            os.makedirs(results_dir, exist_ok=True)

        results_file = '%s/%i.npy' % (results_dir, job_id)
        if os.path.isfile(results_file) and plot:
            print("result already found")
        else:
            params_file = '%s/%i.json' % (results_dir, job_id)
            with open(params_file, 'w') as pf:
                pf.write(json.dumps(params_copy, indent=4))

            if verbose:
                print("...running run_experiment.py with:", protocol, experiment_name)
            prepare_environment(meta)

            # Load Agents Based on Model
            agents = []
            for agent_key in agent_keys:
                agent_params = params.pop(agent_key)
                agents += [Agent(agent_dict=agent_params, name=agent_key, verbose=verbose)]
            params['agents'] = agents

            # Load Protocol and Train (Results callback will collect results)
            module_name = 'protocols.%s.train' % (protocol)
            train = getattr(import_module(module_name), 'train')

            info, results = train(**params,
                                  verbose=verbose,
                                  plot_callback=lambda **kwargs: None)

            # AFTER DONE TRAINING SAVE RESULTS FILE
            results.insert(0, {'protocol': protocol,
                               'trial_num': trial_num,
                               'experiment_name': experiment_name,
                               **info})
            np.save(results_file, results)
            if verbose:
                print("...params for this job have been saved into:", params_file)
                print("...results for this job have been saved into:", results_file)
        # pr.disable()
        # pr.dump_stats('%s%i.pstat'% (experiment_name,job_id) )
        if plot:
            from importlib import util
            if util.find_spec('matplotlib') is not None:
                from plot_experiment import animated_plot
                animated_plot(results=results)
            else:
                print("Cannot plot; matplotlib not found")
    return ()

def main():
    import argparse, textwrap
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    # In outer section of code
    parser = MyParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='python run_experiment.py',
        epilog=textwrap.dedent('''\
            TRY THIS:
            python run_experiment.py\\ 
                --jobs_file=./experiments/shared_preamble/QPSK_neural_vs_clone/jobs.json\\ 
                --job_id=0\\
         '''))

    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--jobs_file", required=True)
    requiredNamed.add_argument("--job_id", type=int, required=False, default=None)
    parser.add_argument("--plot", required=False, action='store_true')
    parser.add_argument("--echo_symlink_to", required=False, default=None)
    parser.add_argument("--job_date", required=False, type=str, default=None)
    args = parser.parse_args()
    run(jobs_file=args.jobs_file, job_id=args.job_id, plot=args.plot, echo_symlink_to=args.echo_symlink_to, job_date=args.job_date)


if __name__ == '__main__':
    try:
        main()
    except AssertionError or Exception:
        import sys, traceback
        traceback.print_exc()
        sys.exit(3)
