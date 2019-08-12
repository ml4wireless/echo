# echo

Lost? Try running: 

```./runecho```

When prompted, 
- you are not on the BRC
- you do not have a jobs.json
- you want to run a script: `scripts/single`
- yes it should have made and returned the corrent jobs.json file
- run one job for job id = 0

# Terminology

**protocol** - the information protocol (i.e. `gradient_passing`,`loss_passing`,`shared_preamble`,`private_preamble`) 

**models** - the model used to learn with (`classic`, `neural`, `poly`)

**mod_order** - `QPSK`,`8PSK`,`QAM16` for 2, 3, or 4 bits per symbol

**experiment** - protocol + mod_order + agent model(s) 
Examples: `experiments/gradient_passing/QPSK_neural_and_classic/` or `experiments/shared_preamble/QPSK_neural_vs_clone/`

**trial** - an instantiation of a single experiment via sampling and generating new seeds

**job** - same as trial but agnostic to the experiment because you can have trials of multiple experiments in one jobs.json file

# Code Overview

`./experiments/[protocol]/model_params.libsonnet` stores the default model parameters for running an experiment under that protocol

`./experiments/create_experiment.py` is used to create an experiment with default settings or command-line specified settings. (See examples of usage in `./scripts/single` and `./scripts/singlecustom`.)

`./experiments/make_jobs.py` is used to create a jobs.json file containing the parameters to run multiple trials (sampling or generating seeds) for a single experiment. *If you supply a file with multiple experiments (one per line), a single jobs.json will be made containing jobs for all of the experiments listed.* (See examples of usage in `./scripts/single`, `./scripts/singlecustom`, `./utils/preprocess_experiments.py`.)

`./utils/preprocess_experiments.py` creates and makes jobs for ALL of the default experiments. Or for a specific protocol. (See examples of usage in `./scripts/singleprotocol` and `./scripts/all`.)

`./run_experiment.py` is used to run experiments from a jobs.json file.

`./plot_experiment.py` is used to plot the results of a single job output.

# Useful commands
`./utils/clean` : cleans out results, experiments and temporary files. 

`./utils/clrbrc` : cleans out outputs from running on brc

`./scripts/single` : see the file. creates 1 experiment using default settings, makes the jobs, and then runs the first job

`./scripts/singlecustom` : see the file. creates 1 experiment using custom params in the custom_params folder, makes the jobs, and then runs the first job

`./scripts/singleprotocol`: see the file, creates the default experiments for a single protocol, makes jobs into a single jobs.json file, runs the first job.

`./scripts/all`: see the file, creates the default experiments for all protocols, makes jobs into a single jobs.json file, runs the first job.

`./runecho`: command-line helper for running experiments. 
