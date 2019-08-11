import os, sys, json, textwrap
# from shutil import copyfile
import json

CWD = os.getcwd()

order_to_bps = {
    'QPSK': 2,
    '8PSK': 3,
    'QAM16': 4
}

default_batch_sizes = {
    'QPSK': 32,  # 256
    '8PSK': 64,  # 64
    'QAM16': 128,  # 128
}

default_num_iterations = {
    'gradient_passing': {
        'QPSK': 150,
        '8PSK': 2000,
        'QAM16': 5000,
    },
    'loss_passing': {
        'QPSK': 800,
        '8PSK': 2000,
        'QAM16': 5000,
    },

    'shared_preamble': {
        'QPSK': 5000,
        '8PSK': 6000,
        'QAM16': 8000,
    },
    'private_preamble': {
        'QPSK': 10000,  # 6000,
        '8PSK': 10000,  # 6000,
        'QAM16': 20000,
    },
}

all_SNR_dbs = {
    'QPSK': [13.0, 8.4, 4.2],
    '8PSK': [18.2, 13.2, 8.4],
    'QAM16': [20.0, 15.0, 10.4]
};

EXPERIMENTS_DIR = os.path.dirname(os.path.realpath(__file__))


def create_experiment(
        protocol,
        mod_order,
        mod1,
        demod1,
        mod2,
        demod2,

        num_trials=5,
        num_logs=20,
        train_snr_db="all",

        signal_power=1.0,
        optimizer='adam',

        experiment_name=None,
        batch_size=None,
        num_iterations=None,
        mod1_param_key=None,
        demod1_param_key=None,
        mod2_param_key=None,
        demod2_param_key=None,
        mod1_param_json=None,
        demod1_param_json=None,
        mod2_param_json=None,
        demod2_param_json=None,
        early_stopping=False,

        delete=False,
        verbose=False,
):
    # model_mod/demod_order
    # make the folder 
    mod_order = mod_order.upper()
    optimizer = optimizer.lower()
    protocol = protocol.lower()
    if protocol in ["gradient_passing", "loss_passing"]:
        mod2 = None
        demod2 = None
    mod1 = mod1.lower()
    mod1_param_key = mod1_param_key if mod1_param_key else '%s_mod_%s' % (mod1, mod_order.lower())
    mod2 = mod2.lower() if mod2 else None
    mod2_param_key = mod2_param_key if mod2_param_key else '%s_mod_%s' % (mod1, mod_order.lower())
    demod1 = demod1.lower() if demod1 else mod1
    demod1_param_key = demod1_param_key if demod1_param_key else '%s_demod_%s' % (demod1, mod_order.lower())
    demod2 = demod2.lower() if demod2 else mod2
    demod2_param_key = demod2_param_key if demod2_param_key else '%s_demod_%s' % (demod2, mod_order.lower())
    num_iterations = num_iterations if num_iterations else default_num_iterations[protocol][mod_order]
    assert train_snr_db in ["high", "mid", "low", "all"]

    if train_snr_db == "all":
        SNR_dbs = all_SNR_dbs[mod_order]
    elif train_snr_db == "high":
        SNR_dbs = [all_SNR_dbs[mod_order][0]]
    elif train_snr_db == "mid":
        SNR_dbs = [all_SNR_dbs[mod_order][1]]
    else:  # low
        SNR_dbs = [all_SNR_dbs[mod_order][2]]

    if not experiment_name:
        if protocol in ['gradient_passing', 'loss_passing']:
            experiment_name = "_".join([mod_order, mod1, 'and', demod1])
        else:
            agent1 = mod1 if mod1 == demod1 else mod1 + "_and_" + demod1
            agent2 = mod2 if mod2 == demod2 else mod2 + "_and_" + demod2
            experiment_name = "_".join([mod_order, agent1, 'vs', agent2])

    # make experiment folder
    experiment_dir = EXPERIMENTS_DIR + '/' + protocol + '/' + experiment_name
    if os.path.isdir(experiment_dir):
        import shutil
        shutil.rmtree(experiment_dir)
        print("Overwriting old dir...", end='')
    os.makedirs(experiment_dir, exist_ok=True)

    if delete:
        import shutil
        shutil.rmtree(experiment_dir)
        print("Deleted %s" % experiment_dir)
        return

    param_jsons = [] #load in param_jsons that have been supplied
    for json_file in [mod1_param_json, demod1_param_json, mod2_param_json, demod2_param_json]:
        param_json = None
        if json_file is not None:
            json_file = os.path.join(CWD, json_file)
            with open(json_file, 'r') as f:
                try:
                    param_json = json.load(f)
                except RuntimeError:
                    print("Error loading json format for: %s"%json_file)

        param_jsons += [param_json]

    mod1_param_json, demod1_param_json, mod2_param_json, demod2_param_json = param_jsons

    # create the experiment dictionary
    experiment_dict = {'num_trials': num_trials, 'train_SNR_dbs': SNR_dbs, 'base': {
        '__meta__': {
            'protocol': protocol,
            'experiment_name': experiment_name,
            'mod_order': mod_order,
            'random_seed': "placeholder",
            'numpy_seed': "placeholder",
            'torch_seed': "placeholder",
            'verbose': bool(verbose),

        },
        'early_stopping': early_stopping,
        'optimizer' if protocol == 'gradient_passing' else None: "$opt$",
        'test_batch_size': 100000,
        'test_SNR_db_type': 'ber_roundtrip',
        'bits_per_symbol': "$bps$",
        'batch_size': batch_size if batch_size else default_batch_sizes[mod_order],
        'num_iterations': num_iterations,
        'results_every': num_iterations // num_logs,
        'signal_power': "$signal_power$",
        None: None,
    }}

    del experiment_dict['base'][None]
    # add_to_params = "{'bits_per_symbol' : bps,  'max_amplitude': signal_power , 'optimizer': %s}"%('null' if protocol == 'gradient_passing' else 'opt')
    classic_params = {}  # {'bits_per_symbol': "$bps$", 'max_amplitude': "$signal_power$"}
    if mod1 and demod1:
        experiment_dict['base']['agent1'] = {
            'bits_per_symbol': "$bps$",
            'max_amplitude': "$signal_power$",
            'optimizer': "$null$" if protocol == 'gradient_passing' else "$opt$",
            'mod_model': mod1,
            'mod_params': classic_params if mod1 == 'classic' else mod1_param_json if mod1_param_json is not None
            else "$params['%s']$" % (mod1_param_key),
            'demod_model': demod1,
            'demod_params': classic_params if demod1 == 'classic' else demod1_param_json if demod1_param_json is not None
            else "$params['%s']$" % (demod1_param_key),
        }

    if mod2 and demod2:
        experiment_dict['base']['agent2'] = {
            # 'bits_per_symbol': order_to_bps[mod_order],
            'bits_per_symbol': "$bps$",
            'max_amplitude': "$signal_power$",
            'optimizer': "$null$" if protocol == 'gradient_passing' else "$opt$",
            'mod_model': mod2,
            'mod_params' if mod2 not in ['clone', 'selfalien'] else None:
                classic_params if mod2 == 'classic' else mod2_param_json if mod2_param_json is not None
                else "$params['%s']$" % (mod2_param_key),
            'demod_model': demod2,
            'demod_params' if demod2 not in ['clone', 'selfalien'] else None:
                classic_params if demod2 == 'classic' else demod2_param_json if demod2_param_json is not None
                else "$params['%s']$" % (demod2_param_key),
            'activation_fn_hidden' if 'selfalien' in [mod2, demod2] else None: 'relu',
            # 'max_amplitude': signal_power,
        }
        del experiment_dict['base']['agent2'][None]

    with open("%s/experiment_params.jsonnet" % experiment_dir, 'w') as file:
        # print("from experiments.%s.model_params import params"%(protocol), file=file)
        # print("params = ", file=file ,end='')
        print("local params = import '../model_params.libsonnet';", file=file)
        print("local bps = %s;" % order_to_bps[mod_order], file=file)
        print("local opt = '%s';" % optimizer, file=file)
        print("local signal_power = %s;" % signal_power, file=file)
        file.write(json.dumps(experiment_dict, indent=4).replace('\"$', '').replace('$\"', ''))

    print("Created experiment params at %s/experiment_params.jsonnet" % experiment_dir)


def main(argv):
    # generate_experiment("shared_preamble", "8PSK","neural","neural","classic", "classic")

    import argparse
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    parser = MyParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        prog='python create_experiment_params.py',
        epilog=textwrap.dedent('''\
            TRY THIS:
            python create_experiment_params.py\\
                --protocol shared_preamble\\
                --mod_order qpsk\\
                --mod1 neural\\
                --demod1 neural\\
                --mod2 clone\\
                --demod2 clone\\
                --num_trials 10\\
                --train_snr_db mid
         '''))
    requiredNamed = parser.add_argument_group('required named arguments')
    importantNamed = parser.add_argument_group('important named arguments')

    requiredNamed.add_argument("--protocol",
                               choices=['gradient_passing', 'loss_passing', 'shared_preamble', 'private_preamble'],
                               required=True)
    requiredNamed.add_argument("--mod_order",
                               help='QPSK (2 bits per symbol), 8PSK (3 bits per symbol), or QAM16 (4 bits per symbol)',
                               choices=['qpsk', '8psk', 'qam16'], required=True)

    requiredNamed.add_argument("--mod1", choices=['classic', 'neural', 'poly'], required=True)

    importantNamed.add_argument("--demod1", choices=['classic', 'neural', 'poly'],
                                help="Optional, (defaults to --mod1)")
    importantNamed.add_argument("--mod2", choices=['classic', 'neural', 'poly', 'clone', 'selfalien'],
                                help="Optional, (defaults to None)")
    importantNamed.add_argument("--demod2", choices=['classic', 'neural', 'poly', 'clone', 'selfalien'],
                                help="Optional, (defaults to --mod2)")

    importantNamed.add_argument("--num_trials", type=int, default=5, help='(default: %(default)s)')
    importantNamed.add_argument("--train_snr_db",
                                help="high is .01%% BER, mid is 1%% BER, low is 10%% BER \n (default: %(default)s)",
                                choices=['high', 'mid', 'low', 'all'], default='all')

    parser.add_argument("--experiment_name", help='optional, custom name for experiment', required=False)
    parser.add_argument("--num_results_logged", type=int, default=20, help='(default: %(default)s)')
    parser.add_argument("--signal_power", type=float, default=1.0, help='(default: %(default)s)')
    parser.add_argument("--optimizer", choices=['adam', 'sgd'], default='adam', help='(default: %(default)s)')
    parser.add_argument("--batch_size", type=int, help='optional, based on --mod_order')
    parser.add_argument("--num_iterations", type=int, help='optional, based on --mod_order and --protocol')
    parser.add_argument("--mod1_param_key", help='optional, can construct from --mod1 and --mod_order', required=False)
    parser.add_argument("--demod1_param_key", help='optional, can construct from --demod1 and --mod_order',
                        required=False)
    parser.add_argument("--mod2_param_key", help='optional, can construct from --mod2 and --mod_order', required=False)
    parser.add_argument("--demod2_param_key", help='optional, can construct from --demod2 and --mod_order',
                        required=False)

    parser.add_argument("--mod1_param_json", help='optional, can construct from --mod1 and --mod_order', required=False)
    parser.add_argument("--demod1_param_json", help='optional, can construct from --demod1 and --mod_order',
                        required=False)
    parser.add_argument("--mod2_param_json", help='optional, can construct from --mod2 and --mod_order', required=False)
    parser.add_argument("--demod2_param_json", help='optional, can construct from --demod2 and --mod_order',
                        required=False)
    parser.add_argument('--delete', action='store_true', help="deletes experiment instead of creating")
    parser.add_argument('--verbose', action='store_true', help="verbose")
    parser.add_argument('--early_stopping', action='store_true',
                        help="agents will stop training early if they are close to optimal")

    args = parser.parse_args()

    if args.protocol in ['shared_preamble', 'private_preamble']:
        if args.mod2 is None:
            parser.error("%s experiments requires --mod1 and --mod2" % args.protocol)
        if args.mod2 == 'selfalien' and args.mod1 != 'neural':
            parser.error("--mod1 must be 'neural' to have --mod2='selfalien'")
        if args.demod2 == 'selfalien' and args.demod1 != 'neural':
            parser.error("--demod1 must be 'neural' to have --demod2='selfalien'")
    else:
        if args.mod2 or args.demod2:
            print("%s experiments do not use --mod2, and --demod2. Ignoring those arguments" % args.protocol)

    if args.mod1_param_key and args.mod1_param_json:
        print("using --mod1_param_json and not --mod1_param_key")
    if args.demod1_param_key and args.demod1_param_json:
        print("using --demod1_param_json and not --demod1_param_key")
    if args.mod2_param_key and args.mod2_param_json:
        print("using --mod2_param_json and not --mod2_param_key")
    if args.demod2_param_key and args.demod2_param_json:
        print("using --demod2_param_json and not --demod2_param_key")

    create_experiment(args.protocol,
                      args.mod_order,
                      args.mod1,
                      args.demod1,
                      args.mod2,
                      args.demod2,

                      num_trials=args.num_trials,
                      num_logs=args.num_results_logged,
                      train_snr_db=args.train_snr_db,

                      signal_power=1.0,
                      optimizer='adam',

                      experiment_name=args.experiment_name,
                      batch_size=args.batch_size,
                      num_iterations=args.num_iterations,
                      early_stopping=args.early_stopping,

                      mod1_param_key=args.mod1_param_key,
                      demod1_param_key=args.demod1_param_key,
                      mod2_param_key=args.mod2_param_key,
                      demod2_param_key=args.demod2_param_key,
                      mod1_param_json=args.mod1_param_json,
                      demod1_param_json=args.demod1_param_json,
                      mod2_param_json=args.mod2_param_json,
                      demod2_param_json=args.demod2_param_json,

                      delete=args.delete,
                      verbose=args.verbose,
                      )


if __name__ == '__main__':
    try:
        main(sys.argv)
    except AssertionError or Exception:
        import sys, traceback
        traceback.print_exc()
        sys.exit(3)