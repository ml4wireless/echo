local params = import '../model_params.libsonnet';
local bps = 2;
local opt = 'adam';
local signal_power = 1.0;
{
    "num_trials": 1,
    "train_SNR_dbs": [
        8.4
    ],
    "base": {
        "__meta__": {
            "protocol": "shared_preamble",
            "experiment_name": "QPSK_neural_hyperparam_search",
            "mod_order": "QPSK",
            "random_seed": "placeholder",
            "numpy_seed": "placeholder",
            "torch_seed": "placeholder",
            "verbose": false
        },
        "test_batch_size": 100000,
        "test_SNR_db_type": "ber_roundtrip",
        "bits_per_symbol": bps,
        "batch_size": 32,
        "num_iterations": 5000,
        "results_every": 250,
        "signal_power": signal_power,
        "agent1": {
            "bits_per_symbol": bps,
            "max_amplitude": signal_power,
            "optimizer": opt,
            "mod_model": "neural",
            "mod_params": {
                "hidden_layers": [
                    50
                ],
                "activation_fn_hidden": "tanh",
                "stepsize_mu": {
                    "sample": "Uniform",
                    "min_val": 0.001,
                    "max_val": 0.0011
                },
                "stepsize_sigma": {
                    "sample": "Uniform",
                    "min_val": 0.0002,
                    "max_val": 0.00021
                },
                "initial_std": 0.2,
                "min_std": 0.01,
                "max_std": 100,
                "lambda_prob": 1e-10,
                "restrict_energy": 1,
                "lambda_p": 0.0,
                "lambda_center": 0.0
            },
            "demod_model": "neural",
            "demod_params": {
                "hidden_layers": [
                    50
                ],
                "activation_fn_hidden": "tanh",
                "loss_type": "l2",
                "stepsize_cross_entropy": {
                    "sample": "Uniform",
                    "min_val": 0.001,
                    "max_val": 0.0011
                },
                "cross_entropy_weight": 1.0
            }
        },
        "agent2": {
            "bits_per_symbol": bps,
            "max_amplitude": signal_power,
            "optimizer": opt,
            "mod_model": "clone",
            "demod_model": "clone"
        }
    }
}