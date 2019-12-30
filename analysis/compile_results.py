# To get directory structure working
import os
import threading
import sys

import numpy as np

ANALYSIS_DIR = os.path.dirname(os.path.realpath(__file__))
print("ANALYSIS_DIR:", ANALYSIS_DIR)
sys.path.append("%s/../utils" % ANALYSIS_DIR)
sys.path.append("%s/../" % ANALYSIS_DIR)


def process_experiment(results_dir, protocol, folder):
    if not os.path.isdir(results_dir):
        return
    # list all files in results
    experiment_results = {
    }
    trials = [f[:-4] for f in os.listdir(results_dir) if ".npy" in f]
    for trial in trials:
        result_npy = os.path.join(results_dir, trial + ".npy")
        print(result_npy + "...")
        # trial_json = os.path.join(results_dir, trial + ".json")
        # with open(trial_json, 'rb') as f:
        #     trial_dict = json.load(f)
        result_array = np.load(result_npy, allow_pickle=True)
        meta, result_array = result_array[0], result_array[1:]
        train_SNR_db = meta['train_SNR_db']
        batch_size = meta['batch_size']
        num_results = meta['num_results']
        test_SNR_dbs = meta.get('test_SNR_dbs', result_array[0]['test_SNR_dbs'])
        training_SNR_dict = experiment_results.get(train_SNR_db,
                                                   {'num_trials': 0,
                                                    'SNR_db_off_for': test_SNR_dbs[4],
                                                    'symbols_sent': [],
                                                    '3db_off': [0.0],
                                                    '5db_off': [0.0],
                                                    'test_SNR_dbs': test_SNR_dbs,
                                                    'max_num_logs': 0
                                                    })
        training_SNR_dict['num_trials'] += 1
        training_SNR_dict['max_num_logs'] = max(training_SNR_dict['max_num_logs'], num_results)
        # In order to deal with early stopping, we subtract if didn't converge,
        # if converged early, then we don't have to fill the array
        db5off = np.ones(training_SNR_dict['max_num_logs'])
        db3off = np.ones(training_SNR_dict['max_num_logs'])
        symbols_sent = training_SNR_dict['symbols_sent']
        for i, result in enumerate(result_array):
            assert i >= len(symbols_sent) or (result['batches_sent'] * batch_size) == symbols_sent[i]
            symbols_sent += [result['batches_sent'] * batch_size] if i >= len(symbols_sent) else []
            db_off = result['db_off'][4]
            if db_off > 5.0:
                db5off[i] -= 1.0
            if db_off > 3.0:
                db3off[i] -= 1.0
        training_SNR_dict['symbols_sent'] = symbols_sent
        # Below handles early stopping
        t5off = training_SNR_dict['5db_off']
        t3off = training_SNR_dict['3db_off']
        for j in range(len(t5off), training_SNR_dict['max_num_logs']):
            t5off = np.append(t5off, t5off[-1])
            t3off = np.append(t3off, t3off[-1])
        training_SNR_dict['5db_off'] = t5off + db5off
        training_SNR_dict['3db_off'] = t3off + db3off

        trial_final_bers = np.array(result_array[-1]['test_bers'])
        trial_final_bers = trial_final_bers[..., np.newaxis]
        if training_SNR_dict.get('final_bers', None) is None:
            training_SNR_dict['final_bers'] = trial_final_bers
        else:
            training_SNR_dict['final_bers'] = np.append(training_SNR_dict['final_bers'], trial_final_bers, axis=1)
        experiment_results[train_SNR_db] = training_SNR_dict
    for k in experiment_results.keys():
        training_SNR_dict = experiment_results[k]
        num_trials = training_SNR_dict['num_trials']
        training_SNR_dict['5db_off'] = training_SNR_dict['5db_off'] / num_trials
        training_SNR_dict['3db_off'] = training_SNR_dict['3db_off'] / num_trials
        sorted_bers = np.sort(training_SNR_dict['final_bers'], axis=1)
        training_SNR_dict['BER_low'] = sorted_bers[:, num_trials * 10 // 100]
        training_SNR_dict['BER_mid'] = sorted_bers[:, num_trials // 2]
        training_SNR_dict['BER_high'] = sorted_bers[:, num_trials * 90 // 100]
    result_file = protocol + "_" + folder + ".npy"
    result_file = os.path.join(ANALYSIS_DIR, "results", result_file)
    print(result_file)
    np.save(result_file, experiment_results)
    return experiment_results


def main():
    # #Job file
    # job_dir = '%s/../experiments/echo_shared_preamble/QAM16_poly_vs_cluster'%D
    os.makedirs(os.path.join(ANALYSIS_DIR, "results"), exist_ok=True)
    for protocol in ['gradient_passing', 'loss_passing', 'shared_preamble', 'private_preamble']:  # 'loss_passing', 'echo_private_preamble','echo_shared_preamble']:
        base_path = "%s/experiments/%s" % (os.path.dirname(ANALYSIS_DIR), protocol)
        print("base path: %s" % base_path)
        dir_list = next(os.walk(base_path))[1]
        # print(job_dir_list)
        threads = []
        for folder in dir_list:
            if 'QPSK' in folder or 'QAM16' in folder or '8PSK' in folder:
                experiment_dir = os.path.join(base_path, folder)
                results_dir = os.path.join(experiment_dir, "results")
                thread = threading.Thread(target=process_experiment, args=(results_dir, protocol, folder))
                thread.start()
                threads.append(thread)
                # experiment_results = process_experiment(results_dir, protocol, folder)
                # result_file = protocol + "_" + folder + ".npy"
                # result_file = os.path.join(ANALYSIS_DIR, "results", result_file)
                # print(result_file)
                # np.save(result_file, experiment_results)
                # np.save(result_file, os.path.join(experiment_dir, "results.npy"))
        for t in threads:
            t.join()

        #                 break
        #         print(job_dir_to_filename("%s/../experiments/gradient_passing/"%D+ job_dir))
        #                 plot_using_results(job_dir)


if __name__ == '__main__':
    main()
