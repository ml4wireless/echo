import os
import sys

ECHO_DIR = os.path.dirname(os.path.realpath(__file__))
assert os.path.basename(ECHO_DIR) == 'echo'

protocol = False
if len(sys.argv) == 2:
    protocol = sys.argv[-1]


###############SO THAT WE DON'T HAVE CONCURRENCY ERRORS###############
suff = ""
i = 0
preprocessout = "%s/_tmp_/preprocess%s.out " % (ECHO_DIR, suff)
while os.path.exists(preprocessout):
    i += 1
    suff = str(i)
    preprocessout = "%s/_tmp_/preprocess%s.out" % (ECHO_DIR, suff)
preprocessexpfolderout = "%s/_tmp_/preprocess_experiment_dirs%s.out" % (ECHO_DIR, suff)

if protocol:
    jobsjson = "%s/experiments/%s/jobs.json" % (ECHO_DIR, protocol)
else:
    jobsjson = "%s/experiments/jobs.json" % (ECHO_DIR)
######################################################################


SNR_SETTING = "all"


def create_params_for_all_experiments():
  experiments = []
  experiments += [['gradient_passing', 'qpsk', i, j, None, None] for i in ["classic", "neural"] for j in
                  ["classic", "neural"] if not (i == 'classic' and j == 'classic')]
  experiments += [['loss_passing', 'qpsk', i, j, None, None] for i in ["neural"] for j in ["classic", "neural"]]
  experiments += [[p, o, 'neural', 'neural', j, j] for p in ["shared_preamble", "private_preamble"] for o in
                  ['qpsk', '8psk', 'qam16'] for j in ["classic", "poly", "clone", "selfalien"]]
  experiments += [[p, o, 'poly', 'poly', 'clone', 'clone'] for p in ["shared_preamble", "private_preamble"] for o in
                  ['qpsk', '8psk', 'qam16']]
  for p, o, m1, d1, m2, d2 in experiments:
        if not protocol or p == protocol and o.upper() == 'QPSK':
            cmd = 'python %s/experiments/create_experiment_params.py ' \
                  '--protocol %s ' \
                  '--mod_order %s ' \
                  '--mod1 %s ' \
                  '--demod1 %s ' \
                  '--num_trials 50 ' \
                  '--train_snr_db %s ' \
                  '  ' % (ECHO_DIR, p, o, m1, d1, SNR_SETTING)
            if m2 and d2:
                cmd += '--mod2 %s ' \
                       '--demod2 %s  ' % (m2, d2)
            print(cmd + "\n", file=open(preprocessout, "a"))
            os.system(cmd)


def make_trials_for_all_experiments():
    experiment_dirs = []
    experiment_dirs += ['%s/experiments/gradient_passing/QPSK_%s_and_%s/' % (ECHO_DIR, i, j) for i in
                        ["classic", "neural"] for j in ["classic", "neural"] if not (i == 'classic' and j == 'classic')]
    experiment_dirs += ['%s/experiments/loss_passing/QPSK_%s_and_%s/' % (ECHO_DIR, i, j) for i in ["neural"] for j in
                        ["classic", "neural"]]
    experiment_dirs += ['%s/experiments/%s/%s_neural_vs_%s/' % (ECHO_DIR, p, o.upper(), j) for p in
                        ["shared_preamble", "private_preamble"] for o in ['qpsk', '8psk', 'qam16'] for j in
                        ["classic", "poly", "clone", "selfalien"]]
    experiment_dirs += ['%s/experiments/%s/%s_poly_vs_clone/' % (ECHO_DIR, p, o.upper()) for p in
                        ["shared_preamble", "private_preamble"] for o in ['qpsk', '8psk', 'qam16']]
    with open(preprocessexpfolderout, 'w') as f:
        for experiment_dir in experiment_dirs:
            if (not protocol or (protocol in experiment_dir)) and 'QPSK' in experiment_dir:
                print(experiment_dir, file=f)
    cmd = 'python %s/experiments/make_jobs.py --experiment_folder=all --file=%s --jobs_json_path=%s' %(ECHO_DIR, preprocessexpfolderout, jobsjson)
    print(cmd, file=open(preprocessout, "a"))
    os.system(cmd)


try:
    print("", end="", file=open(preprocessout, "w"))
    create_params_for_all_experiments()
    print("\n\n\n", file=open(preprocessout, "a"))
    make_trials_for_all_experiments()
    print(jobsjson)
except AssertionError or Exception:
    import sys, traceback
    traceback.print_exc()
    sys.exit(3)

