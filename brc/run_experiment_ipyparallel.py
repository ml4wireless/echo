import getpass
import json
import os
import sys

import numpy as np

ECHO_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(ECHO_DIR)



def execute_parallel(jobs_file, echo_symlink_to=None):
    import ipyparallel as ipp
    client = ipp.Client()
    print('Num clients: %d' % len(client.ids))
    dv = client[:]
    lv = client.load_balanced_view()
    jobs_dispatch = []
    with open(jobs_file) as jfile:
        jobs = json.load(jfile)
    if echo_symlink_to is not None:
        assert os.path.isdir(echo_symlink_to), "Invalid symlink path"
    for job in jobs:
        # DEAL WITH SYMLINKING FOR RUNNING ON BRC
        meta = job['__meta__']
        protocol = meta['protocol']
        experiment_name = meta['experiment_name']
        experiment_dir = os.path.abspath(os.path.join(ECHO_DIR, 'experiments', protocol, experiment_name))
        if not os.path.isdir(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        results_dir = os.path.abspath(os.path.join(experiment_dir, 'results'))
        if echo_symlink_to is not None:
            if not os.path.islink(results_dir):
                if os.path.isdir(results_dir):
                    old_results_dir = os.path.abspath(os.path.join(experiment_dir, 'old_results'))
                    os.makedirs(old_results_dir, exist_ok=True)
                    n = len(os.listdir(old_results_dir))
                    os.rename(results_dir, os.path.abspath(os.path.join(old_results_dir, '%i' % n)))
                _experiment_dir = os.path.abspath(
                    os.path.join(echo_symlink_to, 'experiments', protocol, experiment_name))
                _results_dir = os.path.abspath(os.path.join(_experiment_dir, 'results'))
                if os.path.isdir(_results_dir):
                    _old_results_dir = os.path.abspath(os.path.join(_experiment_dir, 'old_results'))
                    os.makedirs(_old_results_dir, exist_ok=True)
                    n = len(os.listdir(_old_results_dir))
                    os.rename(_results_dir, os.path.abspath(os.path.join(_old_results_dir, '%i' % n)))
                os.makedirs(_results_dir)
                os.symlink(_results_dir, results_dir)
        else:
            os.makedirs(results_dir, exist_ok=True)
        jobs_dispatch += [job]
    print("executing %d jobs" % len(jobs_dispatch))
    # replace these executes with a direct view synchronous mapping to a function that handles imports
    # check this works...this let's us ensure that all imports are completed before running other code
    dv.execute('import torch, random, sys, os, json')
    dv.execute('from importlib import import_module')
    dv.execute('import numpy as np')
    dv.execute('from copy import deepcopy')
    dv.push(dict(ECHO_DIR=ECHO_DIR))
    dv.execute('sys.path.append(\'%s\')' % ECHO_DIR)
    dv.execute('from models.agent import Agent')
    print('sys.path.append(\'%s\')' % ECHO_DIR)
    res = lv.map_sync(client_dispatch, jobs_dispatch)


def client_dispatch(job_description):
    params_copy = deepcopy(job_description)
    meta = job_description.pop('__meta__')
    params = job_description
    keys = params.keys()
    agent_keys = [key for key in keys if 'agent' in key]
    verbose = meta['verbose']
    job_id = meta['job_id']
    trial_num = meta['trial_num']
    protocol = meta['protocol']
    experiment_name = meta['experiment_name']
    experiment_dir = os.path.abspath(os.path.join(ECHO_DIR, 'experiments', protocol, experiment_name))
    results_dir = os.path.abspath(os.path.join(experiment_dir, 'results'))
    print(protocol, experiment_name)

    #PREPARE ENVIRONMENT
    seed = meta.pop("random_seed", 13370)
    numpy_seed = meta.pop("numpy_seed", 1337)
    torch_seed = meta.pop("pytorch_seed", 133)
    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    torch.set_num_threads(1)

    results_file = '%s/%i.npy' % (results_dir, job_id)
    # Load Agents Based on Model
    agents = []
    for agent_key in agent_keys:
        agent_params = params.pop(agent_key)
        agents += [Agent(agent_dict=agent_params, name=agent_key)]
    params['agents'] = agents

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
    np.save(open(results_file, 'wb'), results)
    params_file = '%s/%i.json' % (results_dir, job_id)
    with open(params_file, 'w') as pf:
        pf.write(json.dumps(params_copy, indent=4))
    print("Params for this job have been saved into:")
    print(params_file)
    print("Results for this job have been saved into:")
    print(results_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='ipyparallel echo')
    parser.add_argument(
        '--jobs-json',
        help='The jobs.json file to run',
        required=True
    )
    args = parser.parse_args()
    echo_symlink_dir = '/global/scratch/%s/echo/' % (getpass.getuser())
    assert (os.path.isdir(echo_symlink_dir))
    print("brc/run_experiment_ipyparallel.py begin execute parallel")
    execute_parallel(args.jobs_json, echo_symlink_to=echo_symlink_dir)


if __name__ == '__main__':
    main()
