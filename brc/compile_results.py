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
        result_array = np.load(result_npy)
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
        training_SNR_dict['3db_off'] = training_SNR_dict['5db_off'] / training_SNR_dict['num_trials']
        sorted_bers = np.sort(training_SNR_dict['final_bers'], axis=1)
        num_logs = sorted_bers.shape[1]
        training_SNR_dict['BER_low'] = sorted_bers[:, num_logs * 10 // 100]
        training_SNR_dict['BER_mid'] = sorted_bers[:, num_logs // 2]
        training_SNR_dict['BER_high'] = sorted_bers[:, num_logs * 90 // 100]
    return experiment_results


def ignore():
    symbols_per_iter_dict = {}
    jobs_by_train_SNR_db = {}
    bits_per_symbol_dict = {}
    test_SNR_db_type_dict = {}
    jobs_file = ""
    with open(jobs_file, 'rb') as f:
        jobs_file = []  # pickle.load(f)

    for key in jobs_file:

        #     print(key)
        jobs_dict = jobs_file[key]
        #     print(jobs_dict)
        test_SNR_db_type = jobs_dict['test_SNR_db_type']
        test_SNR_db_type_dict[key] = test_SNR_db_type
        bits_per_symbol = jobs_dict['bits_per_symbol']
        bits_per_symbol_dict[key] = bits_per_symbol
        train_SNR_db = jobs_dict['SNR_db']
        symbols_per_iter = jobs_dict['batch_size']
        symbols_per_iter_dict[key] = symbols_per_iter
        if train_SNR_db in jobs_by_train_SNR_db:
            jobs_by_train_SNR_db[train_SNR_db].append(key)

        else:
            jobs_by_train_SNR_db[train_SNR_db] = [key]
    #     print(train_SNR_db)

    # Go over results
    job_results_dict = {}
    results_dir = '/results/'
    for result_file in os.listdir(results_dir):
        results_dict = np.load(results_dir + result_file).item()
        #     print(result_file)
        #     print(results_dict)
        job_id = int(result_file.split('_')[-1][:-4])
        job_results_dict[job_id] = results_dict

    processed_result_dict = {}
    for job_id in job_results_dict:
        results_dict = job_results_dict[job_id]
        test_SNR_dbs_list = results_dict[0]['test_SNR_dbs']  # Assumes same test SNR dbs for all num iters which is true
        test_SNR_db_type = test_SNR_db_type_dict[job_id]
        bits_per_symbol = bits_per_symbol_dict[job_id]

        # Initialize processed result dict

        test_iters_list = sorted(list(results_dict.keys()))

        total_test_iters = len(list(test_iters_list))
        total_test_SNR_dbs = len(test_SNR_dbs_list)

        processed_result_dict[job_id] = {}

        processed_result_dict[job_id]['error_rate'] = np.zeros((total_test_iters, total_test_SNR_dbs))
        processed_result_dict[job_id]['db_off'] = np.zeros((total_test_iters, total_test_SNR_dbs))

        for idx_num_iters, num_iters in enumerate(test_iters_list):
            cur_results = results_dict[num_iters]

            # Loop over test snr dbs (TODO uncomment this in latest version)
            if test_SNR_db_type == 'ber_roundtrip':

                test_ber_list = cur_results['test_bers']
                for idx_test_SNR_db, test_SNR_db in enumerate(test_SNR_dbs_list):
                    test_ber = test_ber_list[idx_test_SNR_db]
                    optimal_SNR_db = br.get_optimal_SNR_for_BER_roundtrip(test_ber, bits_per_symbol)

                    db_off = test_SNR_db - optimal_SNR_db
                    #                 print(test_SNR_db, optimal_SNR_db, test_ber, db_off)

                    processed_result_dict[job_id]['error_rate'][idx_num_iters, idx_test_SNR_db] = test_ber
                    processed_result_dict[job_id]['db_off'][idx_num_iters, idx_test_SNR_db] = db_off

    ### Fraction of seeds for db off

    fs_db_off = {}
    db_off_list = [0, 0.5, 1, 2, 3, 5, 10]

    for train_SNR_db in jobs_by_train_SNR_db:
        #     print('train_SNR_db', train_SNR_db)

        fs_db_off[train_SNR_db] = {}
        for db_off in db_off_list:
            cnt = 0
            fs_db_off[train_SNR_db][db_off] = np.zeros((total_test_iters, total_test_SNR_dbs))

            for job_id in jobs_by_train_SNR_db[train_SNR_db]:
                if job_id in processed_result_dict:
                    cnt += 1
                    db_off_array = processed_result_dict[job_id]['db_off']
                    fs_db_off[train_SNR_db][db_off] += (db_off_array <= db_off).astype('float')

            if len(jobs_by_train_SNR_db[train_SNR_db]) != 0:
                #             fs_db_off[train_SNR_db][db_off] /= len(jobs_by_train_SNR_db[train_SNR_db])
                if cnt > 0:
                    fs_db_off[train_SNR_db][db_off] /= cnt

        #         print(job_id)
        #         print('db off', db_off)
        #         print(fs_db_off[train_SNR_db][db_off])

    ### Plotting fraction of seeds vs preamble symbols

    #     title = 'Gradient Passing, Neural vs Neural, Bits per symbol: 2'
    #     print("Test SNR dbs: ", test_SNR_dbs_list)

    db_off = 3
    idx_test_SNR_db = 5  # Choose this depending on intended ber: 1: 0, 2: 1e-5, 3:1e-4, 4:1e-3, 5:1e-2, 6:1e-1
    train_SNR_dbs = sorted(list(jobs_by_train_SNR_db.keys()))

    #     print("Train SNR dbs: " , train_SNR_dbs)

    # print(test_iters_list)
    num_preamble_symbols_array = np.array(test_iters_list) * symbols_per_iter_dict[0]

    dict_3db = {}
    # print(train_SNR_dbs)
    # print(fs_db_off)
    for train_SNR_db in train_SNR_dbs:
        if train_SNR_db in fs_db_off:
            fs_y = fs_db_off[train_SNR_db][db_off][:, idx_test_SNR_db]
            fs_x = num_preamble_symbols_array
            #     print(fs_y)
            #     print(fs_x)
            dict_3db[train_SNR_db] = {}
            dict_3db[train_SNR_db]['x'] = fs_x
            dict_3db[train_SNR_db]['y'] = fs_y
            plt.plot(fs_x, fs_y, 'o-', label='train_SNR_db = ' + str(train_SNR_db))

    plt.ylabel('Fraction of seeds')
    plt.xlabel('Number of preamble symbols transmitted')
    plt.legend()
    plt.title('Fraction of seeds within 3db off vs number of preamble symbols transmitted')

    figname = 'fraction_seeds_3db.png'
    plt.savefig(figname, format='png')
    plt.show()

    # Save processed results
    filename = 'fraction_seeds_3db.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump(dict_3db, f)

    ### Plotting fraction of seeds vs preamble symbols

    #     title = 'Gradient Passing, Neural vs Classic, Bits per symbol: 2'
    #     print("Test SNR dbs: ", test_SNR_dbs_list)

    db_off = 5
    idx_test_SNR_db = 5  # Choose this depending on intended ber: 1: 0, 2: 1e-5, 3:1e-4, 4:1e-3, 5:1e-2, 6:1e-1
    train_SNR_dbs = sorted(list(jobs_by_train_SNR_db.keys()))

    #     print("Train SNR dbs: " , train_SNR_dbs)

    num_preamble_symbols_array = np.array(test_iters_list) * symbols_per_iter_dict[0]

    # Store the processed results
    dict_5db = {}

    for train_SNR_db in train_SNR_dbs:
        fs_y = fs_db_off[train_SNR_db][db_off][:, idx_test_SNR_db]
        fs_x = num_preamble_symbols_array
        #         print(fs_y)
        plt.plot(fs_x, fs_y, 'o-', label='train_SNR_db = ' + str(train_SNR_db))

        dict_5db[train_SNR_db] = {}
        dict_5db[train_SNR_db]['x'] = fs_x
        dict_5db[train_SNR_db]['y'] = fs_y

    plt.ylabel('Fraction of seeds')
    plt.xlabel('Number of preamble symbols transmitted')
    plt.legend()
    plt.title('Fraction of seeds within 5db off vs number of preamble symbols transmitted')

    # figname = job_dir_to_figname(job_dir) + 'fraction_seeds_5db.png'
    # plt.savefig(figname, format='png')
    #
    # plt.show()

    # # Save processed results
    # filename = job_dir_to_filename(job_dir) + 'fraction_seeds_5db.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump(dict_5db, f)

    # Preprocessing data for round trip BER graph
    percentile_high = 90
    percentile_low = 10

    BER_low = {}
    BER_mid = {}
    BER_high = {}
    for train_SNR_db in jobs_by_train_SNR_db:
        #     print('train_SNR_db', train_SNR_db)
        BER_low[train_SNR_db] = np.zeros((total_test_iters, total_test_SNR_dbs))
        BER_mid[train_SNR_db] = np.zeros((total_test_iters, total_test_SNR_dbs))
        BER_high[train_SNR_db] = np.zeros((total_test_iters, total_test_SNR_dbs))

        for i in range(total_test_iters):
            for j in range(total_test_SNR_dbs):
                cur_ber_list = []
                for job_id in jobs_by_train_SNR_db[train_SNR_db]:
                    if job_id in processed_result_dict:
                        ber_array = processed_result_dict[job_id]['error_rate']
                        cur_ber_list.append(ber_array[i, j])

                if len(cur_ber_list) > 0:
                    sorted_cur_ber_list = sorted(cur_ber_list)
                    cur_len = len(sorted_cur_ber_list)
                    cur_mid_idx = cur_len // 2
                    cur_high_idx = cur_len * percentile_high // 100
                    cur_low_idx = cur_len * percentile_low // 100

                    cur_mid = sorted_cur_ber_list[cur_mid_idx]
                    cur_high = sorted_cur_ber_list[cur_high_idx]
                    cur_low = sorted_cur_ber_list[cur_low_idx]

                    BER_low[train_SNR_db][i, j] = cur_low
                    BER_high[train_SNR_db][i, j] = cur_high
                    BER_mid[train_SNR_db][i, j] = cur_mid

    test_iter_idx = -1  # For end of  training

    train_SNR_dbs = sorted(list(jobs_by_train_SNR_db.keys()))
    #     print("Train SNR dbs: ", train_SNR_dbs)
    idx_train_SNR_dbs = [-1, -2,
                         -3]  # Choose this depending on intended ber: 0: 0, 1: 1e-5, 2:1e-4, 3:1e-3, 4:1e-2, 5:1e-1
    plot_train_SNR_dbs = [train_SNR_dbs[i] for i in idx_train_SNR_dbs]
    #     print("plot train SNR dbs", plot_train_SNR_dbs)

    # Get baseline
    fs_y_baseline = [br.get_optimal_BER_roundtrip(test_SNR_db, bits_per_symbol) for test_SNR_db in
                     test_SNR_dbs_list[1:]]  # TODO change to round trip

    dict_ber = {'results': {}, 'baseline': fs_y_baseline}

    for train_SNR_db in plot_train_SNR_dbs:
        fs_mid = [BER_mid[train_SNR_db][-1, i + 1] for i in range(len(test_SNR_dbs_list[1:]))]
        fs_low = [BER_low[train_SNR_db][-1, i + 1] for i in range(len(test_SNR_dbs_list[1:]))]
        fs_high = [BER_high[train_SNR_db][-1, i + 1] for i in range(len(test_SNR_dbs_list[1:]))]
        fs_mid = np.array(fs_mid)
        fs_low = np.array(fs_low)
        fs_high = np.array(fs_high)

        fs_x = test_SNR_dbs_list[1:]
        #     plt.plot(fs_x, fs_mid,  'o-', label = 'train_SNR_db = ' + str(train_SNR_db))
        #     plt.fill_between(fs_x, fs_high, fs_low, color = 'lightgreen', alpha = 0.5)

        dict_ber['results'][train_SNR_db] = {}
        dict_ber['results'][train_SNR_db]['fs_x'] = fs_x
        dict_ber['results'][train_SNR_db]['fs_mid'] = fs_mid
        dict_ber['results'][train_SNR_db]['fs_high'] = fs_high
        dict_ber['results'][train_SNR_db]['fs_low'] = fs_low
        plt.errorbar(fs_x, fs_mid, yerr=[fs_mid - fs_low, fs_high - fs_mid], elinewidth=3, fmt='o-',
                     label='train_SNR_db = ' + str(train_SNR_db), )

    plt.plot(fs_x, fs_y_baseline, 'o-b', label='Baseline')
    # print(fs_y_baseline)
    plt.legend(loc='lower left')
    plt.yscale('log')
    plt.xlabel('SNR(db)')
    plt.ylabel('Round trip BER')
    plt.title('Round trip BER vs SNR(db)')
    #
    # figname = job_dir_to_figname(job_dir) + 'BER.png'
    # plt.savefig(figname, format='png')

    plt.show()
    #
    # # Save processed results
    # filename = job_dir_to_filename(job_dir) + 'BER.pkl'
    # with open(filename, 'wb') as f:
    #     pickle.dump(dict_ber, f)

    #     fs_x = num_preamble_symbols_array
    #     print(fs_y)
    #     plt.plot(fs_x, fs_y, 'o-', label = 'train_SNR_db = ' + str(train_SNR_db))


# #Job file
# job_dir = '%s/../experiments/echo_shared_preamble/QAM16_poly_vs_cluster'%D
os.makedirs(os.path.join(BRC_DIR, "results"), exist_ok=True)
for protocol in ['shared_preamble']:  # 'loss_passing', 'echo_private_preamble','echo_shared_preamble']:
    base_path = "%s/../experiments/%s" % (BRC_DIR, protocol)
    dir_list = next(os.walk(base_path))[1]
    print(base_path)
    # print(job_dir_list)
    for folder in dir_list:
        experiment_dir = os.path.join(base_path, folder)
        experiment_results = process_experiment(experiment_dir)
        result_file = os.path.join(BRC_DIR, "results", folder + ".npy")
        np.save(result_file, experiment_results)

    #                 break
    #         print(job_dir_to_filename("%s/../experiments/gradient_passing/"%D+ job_dir))
    #                 plot_using_results(job_dir)
