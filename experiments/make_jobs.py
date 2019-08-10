import os, glob, sys, json, textwrap, importlib
import _jsonnet
import numpy as np
from pprint import pprint
from copy import deepcopy

CWD = os.getcwd()
SEED = 2019

EXPERIMENT_DIR = os.path.dirname(os.path.realpath(__file__))


def make_jobs(list_experiment_params):
    trials = []
    index = 0
    for experiment_params in list_experiment_params:
        rs = np.random.RandomState(SEED)
        training_snrs = experiment_params['train_SNR_dbs']
        # RUN num_trials TRIALS FOR EACH TRAINING SNR
        for train_snr in training_snrs:
            for trial in range(experiment_params['num_trials']):
                trial_params = deepcopy(experiment_params['base'])
                trial_params['__meta__']['trial_num'] = trial + 1
                trial_params['__meta__']['total_trials'] = experiment_params['num_trials']
                trial_params['__meta__']['job_id'] = index
                trial_params['__meta__']['SNR_db'] = train_snr
                trial_params['train_SNR_db'] = train_snr

                # SAMPLE (deep) ANY PARAMETERS THAT REQUIRE IT
                deep_sample(trial_params, rs)

                # PASS ON ANY PARAMETERS IN AGENT TO MOD and DEMOD
                a1 = trial_params.get("agent1")
                inherit(a1)
                # DEAL WITH CLONE AND SELFALIEN
                a2 = trial_params.get("agent2", None)
                if a2:
                    if a2['mod_model'] in ['clone', 'selfalien']:
                        a2['mod_model'] = a1['mod_model']
                        a2['mod_params'] = deepcopy(a1['mod_params'])
                    if a2['demod_model'] in ['clone', 'selfalien']:
                        a2['demod_model'] = a1['demod_model']
                        a2['demod_params'] = deepcopy(a1['demod_params'])
                    # PASS ON ANY PARAMETERS IN AGENT TO MOD and DEMOD
                    inherit(a2)

                # POPULATE (deep) ANY FIELDS NAMED SEEDS WITH A RANDOM VALUE
                set_seeds(trial_params, rs)
                trials += [trial_params]
                index += 1
    return trials


def deep_sample(d, rs):
    for key in sorted(d.keys()):
        val = d[key]
        if isinstance(val, dict):
            if 'sample' in val.keys():
                method = val['sample'].lower()
                if method == 'discrete':
                    options = val['values']
                    d[key] = options[rs.randint(len(options))]
                else:
                    min_val = val['min_val']
                    max_val = val['max_val']
                    assert min_val <= max_val, "sampled params with specified range must have min_val <= max_val"
                    if min_val == max_val:
                        return max_val
                    if method == 'uniform':
                        d[key] = rs.uniform(low=min_val, high=max_val)
                    elif method == 'integer_uniform':
                        d[key] = rs.randint(low=min_val, high=max_val)
                    elif method == 'log_uniform':
                        d[key] = np.exp(rs.uniform(low=np.log(min_val), high=np.log(max_val)))
                    else:
                        assert method in ['discrete', 'uniform', 'integer_uniform', 'log_uniform'], \
                            "sample method not recognized: %s; 'discrete', \
                            'uniform' 'integer_uniform', 'log_uniform' supported" % method
            else:
                deep_sample(val, rs)


def set_seeds(d, rs):
    for key in sorted(d.keys()):
        if 'seed' in key.lower():
            d[key] = rs.randint(100000)
        if isinstance(d[key], dict):
            set_seeds(d[key], rs)


def inherit(a):
    # pprint(a)
    not_inherited_keys = ['mod_model', 'demod_model', 'mod_params', 'demod_params']
    for key in a.keys():
        if key not in not_inherited_keys:
            a['mod_params'][key] = a[key]
            a['demod_params'][key] = a[key]


def main(argv):
    import argparse
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                      epilog=textwrap.dedent('''\
						TRY THIS:
						python make_jobs.py --experiment_folder ./gradient_passing/QPSK_neural_and_neural
						'''))
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument("--experiment_folder", required=True)
    parser.add_argument("--file",
                        help='when --experiment_folder=all supply a file with every experiment_folder name and one '
                             'jobs.json will be created',
                        required=False)
    parser.add_argument("--jobs_json_path",
                        help='saves the jobs file to the path specified',
                        required=False)
    args = parser.parse_args()

    if args.experiment_folder == 'all':
        all_data = []
        with open(args.file, 'r') as fp:
            lines = fp.read().splitlines()
            for line in lines:
                assert os.path.isdir(line), "Folder specified by in %s must exist: %s, %s" % (args.file, line, str(os.path.isdir(line)))
                jsonnet_file = "%s/experiment_params.jsonnet" % line
                data = json.loads(_jsonnet.evaluate_file(jsonnet_file))
                all_data += [data]
        jobs = make_jobs(all_data)
        jobs_json = args.jobs_json_path if args.jobs_json_path else "%s/jobs.json" % (os.path.dirname(args.file))
        with open(jobs_json, 'w') as file:
            file.write(json.dumps(jobs, indent=4))

        print("Made %d jobs for %d experiments specified in %s: %s" % (
            len(jobs), len(lines), args.file, "%s/jobs.json" % (os.path.dirname(args.file))))
    else:
        assert os.path.isdir(
            args.experiment_folder), "Folder specified by --experiment_folder must exist: %s" % args.experiment_folder

        jsonnet_file = "%s/experiment_params.jsonnet" % args.experiment_folder
        data = json.loads(_jsonnet.evaluate_file(jsonnet_file))
        jobs = make_jobs([data])
        jobs_json = args.jobs_json_path if args.jobs_json_path else "%s/jobs.json" % args.experiment_folder
        with open(jobs_json, 'w') as file:
            file.write(json.dumps(jobs, indent=4, sort_keys=False))
        print("Made %d jobs for %s" % (len(jobs), args.experiment_folder))


if __name__ == '__main__':
    try:
        main(sys.argv)
    except AssertionError or Exception:
        import traceback
        traceback.print_exc()
        exit(3)
