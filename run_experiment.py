import os
import sys
import torch, numpy, random
import numpy as np
from models.agent import Agent
# from typing import Union
from importlib import import_module
import json
from pprint import pprint
import matplotlib.pyplot as plt
from utils.util_data import integers_to_symbols, get_grid_2d
from matplotlib.patches import Ellipse
from copy import deepcopy
import cProfile as profile

pr = profile.Profile()

##TODO
# 1. check that make_jobs in a batch  == make_jobs single (seeds are set properly)
# DONE 2. make sure that rerunning the same job is reproduible
# 3. make sure that rerunning the same job but with other stuff in between is reproducible
# 4. check that results here == results in torch echo

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


def animated_plot(results=False,
                  results_file="/Users/caryn/Desktop/echo/experiments/loss_passing/QPSK_neural_and_classic/results/1.npy"):  # (result):
    if not results:
        results = np.load(open(results_file, 'rb'))
    result = results[-1]
    grid = get_grid_2d()
    print(grid.shape, results[0]['demod_grid_1'].shape)
    for result in results:
        m1, c1 = (result['mod_std_1'], result['constellation_1'])
        d1 = result['demod_grid_1']
        m2, c2 = (result['mod_std_2'], result['constellation_2'])
        d2 = result['demod_grid_2']

        i_means = c1[:, 0]
        i_std = m1[0]

        q_means = c1[:, 1]
        q_std = m1[1]

        colors = ['r', 'y', 'g', 'b']

        # print(len(i_means))
        plt.subplot(1, 2, 1, aspect='equal')
        plt.scatter(i_means, q_means, c=['r', 'y', 'g', 'b'], )
        ells = [Ellipse(xy=c1[i], width=np.sqrt(9.210) * m1[0], height=np.sqrt(9.210) * m1[1])
                for i in range(len(c1))]
        ax = plt.gca()
        for i, e in enumerate(ells):
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)
            e.set_alpha(.2)
            e.set_facecolor(colors[i])
        for label, (x,
                    y) in zip(integers_to_symbols(np.array((range(len(c1)))), 2), c1):
            plt.annotate(
                label,
                xy=(x, y), xytext=(0, 0), textcoords='offset points')
            # textcoords='offset points', ha='right', va='bottom',
            # bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            # arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        plt.subplot(1, 2, 2, aspect='equal')

        plt.scatter(grid[:, 0], grid[:, 1], c=[colors[i] for i in d1])
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)

        plt.pause(0.05)
        plt.clf()
    plt.draw()

    # plot(constellation_1) w/

    # pprint(m1)
    # print(d1)
    return


# animated_plot()


def run(jobs_file, job_id=None, plot=False, echo_symlink_to=None):
    # In section you want to profile
    pr.enable()

    with open(jobs_file) as jfile:
        jobs = json.load(jfile)
    if job_id is not None:  # 0 = False you dummy
        jobs = [jobs[job_id]]
    for params in jobs:
        SAVE_PLOTS = False
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

        results_file = '%s/%i_%i.npy' % (results_dir, job_id, trial_num)
        if os.path.isfile(results_file) and plot:
            print("result already found")
        else:
            print(results_file)
            params_file = '%s/%i_%i.json' % (results_dir, job_id, trial_num)
            with open(params_file, 'w') as pf:
                pf.write(json.dumps(params_copy, indent=4))

            print(protocol, experiment_name)
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

            results = train(**params,
                            verbose=verbose,
                            plot_callback=lambda **kwargs: None)

            # AFTER DONE TRAINING SAVE RESULTS FILE
            results.insert(0, {'protocol': protocol, 'trial_num': trial_num, 'experiment_name': experiment_name})
            np.save(open(results_file, 'wb'), results)

        # PLOTTING
        if plot:
            animated_plot(results_file=results_file)
    return ()


# jobs_file = "%s/experiments/private_preamble/QPSK_neural_vs_classic/jobs.json"%ECHO_DIR
# def play():
#     jobs_file = "%s/jobs.json" % ECHO_DIR
#     with open(jobs_file) as jfile:
#         jobs = json.load(jfile)
#     animated_plot(results_file=run(jobs[1050]))


# play()

# def save_plots(batch_size, plot_num):
#     plot_data_si = get_random_data_si(n=batch_size, bits_per_symbol=bits_per_symbol)
#     plot_agent(A,
#                preamble_si=plot_data_si,
#                plots_dir=plots_dir,
#                step=i,
#                plot_count=plot_num,
#                show=False)
#     plot_agent(B,
#                preamble_si=plot_data_si,
#                plots_dir=plots_dir,
#                step=i,
#                plot_count=plot_num,
#                show=False)

#
# def plot_agent(agent, *, preamble_si, plots_dir=None, step=0, plot_count=0, show=False):
#     import sys
#     from utils.visualize import visualize_constellation
#     from utils.visualize import visualize_decision_boundary
#     # import matplotlib
#     # matplotlib.use('Agg')
#     import matplotlib.pyplot as plt
#     # Modulator
#     if (agent.mod.mod_class != 'classic' or plot_count <= 1):
#         data_m = agent.mod.modulate(data_si=preamble_si, mode='explore')
#         data_m_centers = agent.mod.modulate(data_si=preamble_si, mode='exploit')
#         args = {"data": data_m,
#                 "data_centers": data_m_centers,
#                 "labels": preamble_si,
#                 "legend_map": {i: i for i in range(2 ** agent.mod.bits_per_symbol)},
#                 "title_string": 'Modulator %s, %s: Step %d' % (
#                     agent.mod.mod_class,
#                     agent.name,
#                     step),
#                 "show": show}
#         visualize_constellation(**args)
#         if not show:
#             plt.savefig("%s/%s_mod-%d.png" % (plots_dir, "_".join(agent.name.lower().split(" ")), plot_count))
#         plt.close()
#     # Demodulator
#     if (agent.demod.demod_class != 'classic' or plot_count <= 1):
#         args = {"points_per_dim": 100,
#                 "legend_map": {i: i for i in range(2 ** agent.demod.bits_per_symbol)},
#                 "title_string": 'Demodulator %s, %s: Step %d' % (
#                     agent.demod.demod_class,
#                     agent.name,
#                     step),
#                 "show": show}
#         visualize_decision_boundary(agent.demod, **args)()
#         if not show:
#             plt.savefig("%s/%s_demod-%d.png" % (plots_dir, "_".join(agent.name.lower().split(" ")), plot_count))
#         plt.close


def main(argv):
    import argparse, textwrap
    class MyParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write('error: %s\n' % message)
            self.print_help()
            sys.exit(2)

    # In outer section of code

    pr.disable()
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
    args = parser.parse_args()
    run(jobs_file=args.jobs_file, job_id=args.job_id, plot=args.plot, echo_symlink_to=args.echo_symlink_to)
    # code of interest
    pr.disable()

    # Back in outer section of code
    pr.dump_stats('profile.pstat')


if __name__ == '__main__':
    try:
        main(sys.argv)
    except AssertionError or Exception:
        import sys, traceback

        traceback.print_exc()
        sys.exit(3)
