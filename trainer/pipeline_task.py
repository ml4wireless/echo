#!/usr/bin/env python # pylint: disable=g-unknown-interpreter

import json
import os
import sys
import random
from importlib import import_module
import numpy as np
import torch
TRAINER_DIR = os.path.dirname(os.path.realpath(__file__))
ECHO_DIR = os.path.dirname(TRAINER_DIR)
sys.path.append(ECHO_DIR)
from models.agent import Agent

RESULT_FILE = os.environ['RESULT_FILE']
INDEX = os.environ['INDEX']




def prepare_environment(params):
    seed = params.pop("random_seed", 13370)
    numpy_seed = params.pop("numpy_seed", 1337)
    torch_seed = params.pop("pytorch_seed", 133)

    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        np.random.seed(numpy_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    torch.set_num_threads(1)
    sys.path.append(ECHO_DIR)


def run(params):
    keys = params.keys()
    agent_keys = [key for key in keys if 'agent' in key]
    meta = params.pop('__meta__')
    verbose = meta['verbose']
    trial_num = meta['trial_num']
    protocol = meta['protocol']
    experiment_name = meta['experiment_name']
    prepare_environment(meta)
    # Load Agents Based on Model
    agents = []
    for agent_key in agent_keys:
        agent_params = params.pop(agent_key)
        agents += [Agent(agent_dict=agent_params, name=agent_key)]
    params['agents'] = agents

    # Load Protocol and Train (Results callback will collect results)
    module_name = 'protocols.%s.train' % (protocol)
    train = getattr(import_module(module_name), 'train')

    info, results = train(**params,
                          verbose=False)

    # AFTER DONE TRAINING SAVE RESULTS FILE
    results.insert(0, {'experiment_name': experiment_name,
                       'protocol': protocol,
                       'trial_num': trial_num,
                       **info})
    np.save(RESULT_FILE, results)


def main():
    with open('%s/work/%s.json'%(TRAINER_DIR, INDEX), 'r') as file:
        params = json.load(file)
    run(params)


if __name__ == '__main__':
    main()
