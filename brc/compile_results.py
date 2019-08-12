# To get directory structure working
import sys
import os
import json
import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os

BRC_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../utils" % BRC_DIR)
sys.path.append("%s/../" % BRC_DIR)
import utils.util_lookup_table

br = utils.util_lookup_table.BER_lookup_table()


def process_experiment(experiment_dir):
    print("experiment dir: ", experiment_dir)
    results_dir = os.path.join(experiment_dir, "results")
    if not os.path.isdir(results_dir):
        return
    # list all files in results
    experiment_results = {
    }
    jobs = [f[:-4] for f in os.listdir(results_dir) if ".npy" in f]

    for job in jobs:
        trial_num = job.split("_")[-1]
        result_npy = os.path.join(results_dir, job + ".npy")
        job_json = os.path.join(results_dir, job + ".json")
        with open(job_json, 'rb') as f:
            job_dict = json.load(f)
        num_iterations = job_dict['num_iterations']
        results_every = job_dict['results_every']
        batch_size = job_dict['batch_size']
        bits_per_symbol = job_dict['bits_per_symbol']
        train_SNR_db = job_dict["SNR_db"]
        result_array = np.load(result_npy, allow_pickle=True)
        num_results = len(result_array)
        training_SNR_dict = experiment_results.get(train_SNR_db, {'num_trials': 0})
        training_SNR_dict['num_trials'] += 1
        if training_SNR_dict.get('symbols_sent', None) is None:
            training_SNR_dict['iterations'] = [i * results_every for i in range((num_iterations // results_every) + 1)]
            training_SNR_dict['symbols_sent'] = [i * results_every * 2 * batch_size for i in
                                                 range((num_iterations // results_every) + 1)]
            training_SNR_dict['3db_off'] = np.zeros([num_results], dtype=np.float32)
            training_SNR_dict['5db_off'] = np.zeros([num_results], dtype=np.float32)
        else:
            # print(training_SNR_dict['3db_off'])
            assert num_results == len(
                training_SNR_dict['3db_off']), "something went terribly wrong, trials not same num iters"
        for i, result in enumerate(result_array):
            # other result keys: 'test_sers' 'mod_std_1' 'constellation_1' 'demod_grid_1'
            # 'mod_std_2' 'constellation_2' 'demod_grid_2'
            test_SNR_dbs = result['test_SNR_dbs']
            test_bers = np.mean([result['test_bers'][0], result['test_bers'][1]], axis=0)
            # print(test_bers)
            if not training_SNR_dict.get('test_SNR_dbs', False):
                training_SNR_dict['test_SNR_dbs'] = test_SNR_dbs[1:]
                training_SNR_dict['convergence_for'] = test_SNR_dbs[5]
            db_off = test_SNR_dbs[5] - br.get_optimal_SNR_for_BER_roundtrip(test_bers[5], bits_per_symbol)
            if db_off <= 5.0:
                training_SNR_dict['5db_off'][i] += 1.0
            if db_off <= 3.0:
                training_SNR_dict['3db_off'][i] += 1.0
        trial_final_bers = np.mean([result_array[-1]['test_bers'][0], result_array[-1]['test_bers'][1]], axis=0)
        trial_final_bers = trial_final_bers[1:].reshape((len(test_SNR_dbs) - 1, 1))  # last result of your trial;
        # but ignore first BER b/c 1st result is testing @ the train snr
        if training_SNR_dict.get('final_bers', None) is None:
            # for n = num of testing SNRs - 1 (see note above)
            # collect all the final bers of all trials then at the end, compute the quartiles
            training_SNR_dict['final_bers'] = trial_final_bers
        else:
            # print("before", training_SNR_dict['final_bers'].shape)
            training_SNR_dict['final_bers'] = np.append(training_SNR_dict['final_bers'], trial_final_bers, axis=1)
            # print("after", training_SNR_dict['final_bers'].shape)
        experiment_results[train_SNR_db] = training_SNR_dict
    # average convergence by num_trials
    for k in experiment_results:
        training_SNR_dict = experiment_results[k]
        training_SNR_dict['5db_off'] = training_SNR_dict['5db_off'] / training_SNR_dict['num_trials']
        training_SNR_dict['3db_off'] = training_SNR_dict['3db_off'] / training_SNR_dict['num_trials']
        sorted_bers = np.sort(training_SNR_dict['final_bers'], axis=1)
        num_logs = sorted_bers.shape[1]
        training_SNR_dict['BER_low'] = sorted_bers[:, num_logs * 10 // 100]
        training_SNR_dict['BER_mid'] = sorted_bers[:, num_logs // 2]
        training_SNR_dict['BER_high'] = sorted_bers[:, num_logs * 90 // 100]
    return experiment_results

# #Job file
# job_dir = '%s/../experiments/echo_shared_preamble/QAM16_poly_vs_cluster'%D
os.makedirs(os.path.join(BRC_DIR, "results"), exist_ok=True)
for protocol in ['shared_preamble']:  # 'loss_passing', 'echo_private_preamble','echo_shared_preamble']:
    base_path = "%s/../experiments/%s" % (BRC_DIR, protocol)
    dir_list = next(os.walk(base_path))[1]
    print(base_path)
    # print(job_dir_list)
    for folder in dir_list:
        if 'QPSK' in folder and 'hyper' not in folder:
            experiment_dir = os.path.join(base_path, folder)
            experiment_results = process_experiment(experiment_dir)
            result_file = os.path.join(BRC_DIR, "results", folder + ".npy")
            np.save(result_file, experiment_results)

    #                 break
    #         print(job_dir_to_filename("%s/../experiments/gradient_passing/"%D+ job_dir))
    #                 plot_using_results(job_dir)
