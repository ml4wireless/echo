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
from google.cloud import storage

def save_file(job_dir, file):
    # Example: job_dir = 'gs://BUCKET_ID/hptuning_sonar/1'
    job_dir = job_dir.replace('gs://', '')  # Remove the 'gs://'
    # Get the Bucket Id
    bucket_id = job_dir.split('/')[0]
    # Get the path. Example: 'hptuning_sonar/1'
    bucket_path = job_dir.lstrip('{}/'.format(bucket_id))

    # Upload the model to GCS
    bucket = storage.Client().bucket(bucket_id)
    blob = bucket.blob('{}/{}'.format(
        bucket_path,
        file))
    blob.upload_from_filename(file)



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


def run(params, task_id, job_dir):
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
    result_file = '%i.npy'%task_id
    np.save(result_file, results)
    save_file(job_dir, result_file)


def main(args):
    with open('%s/work/%s.json'%(TRAINER_DIR, args.task_id), 'r') as file:
        params = json.load(file)
    run(params, args.task_id, args.job_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--job-dir', type=str)
    args = parser.parse_args()
    main(args)
