import json
def execute_parallel(jobs_file, echo_symlink_to=None):
    import :ipyparallel as ipp
    client = ipp.Client()
    print('Num clients: %d' % len(client.ids))
    dv = client[:]
    lv = client.load_balanced_view()
    jobs_dispatch = []
    with open(jobs_file) as jfile:
        jobs = json.load(jfile)

def client_dispatch(job_description):
    import importlib
    import numpy as np
    experiment_name, experiment_setting_name, results_dir, job_id, job_params, save_plots, plots_dir, verbose = job_description
    module_name = 'experiments.%s.main' % experiment_name
    print("client running: ",module_name, job_id)
    experiment = importlib.import_module(module_name)
    #If want to save plots, can use these
    job_params['save_plots'] = save_plots
    job_params['plots_dir'] = plots_dir
    job_params['job_id'] = job_id
    job_params['verbose'] = verbose
    res = experiment.run(**job_params)
    np.save("%s/result_%s" % (results_dir, job_id), res)


    keys = params.keys()
    agent_keys = [key for key in keys if 'agent' in key]
    meta = params.pop('__meta__')
    verbose = meta['verbose']
    job_id = meta['job_id']
    trial_num = meta['trial_num']
    protocol = meta['protocol']
    experiment_name = meta['experiment_name']
    experiment_dir = os.path.abspath(os.path.join(ECHO_DIR, 'experiments', protocol, experiment_name))
    protocol_dir = os.path.abspath(os.path.join(ECHO_DIR, 'protocols', protocol))

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

    return 'experiments/%s/%s---%d'%(experiment_name, experiment_setting_name, job_id)








def run(jobs_file, job_id=None, plot=False, echo_symlink_to=None):
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
        protocol_dir = os.path.abspath(os.path.join(ECHO_DIR, 'protocols', protocol))

        results_dir = os.path.abspath(os.path.join(experiment_dir, 'results'))
        #DEAL WITH SYMLINKING FOR RUNNING ON BRC
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
            # params_file = '%s/%i_%i.json' % (results_dir, job_id, trial_num)
            # with open(params_file, 'w') as pf:
            #     pf.write(json.dumps(params_copy, indent=4))

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
            module_name = 'protocols.%s.evaluate' % (protocol)
            evaluate = getattr(import_module(module_name), 'evaluate')

            class Results():
                def __init__(self, func):
                    self.results = []
                    self.func = func

                def callback(self, **kwargs):
                    self.results += [self.func(**kwargs)]

                def make_callback(self):
                    return self.callback

            evaluation = Results(evaluate)
            train(**params,
                  verbose=verbose,
                  evaluate_callback=evaluation.make_callback(),
                  plot_callback=lambda **kwargs: None)

            # AFTER DONE TRAINING SAVE RESULTS FILE
            np.save(open(results_file, 'wb'), evaluation.results)

        # PLOTTING
        if plot:
            animated_plot(results_file=results_file)
    return ()
