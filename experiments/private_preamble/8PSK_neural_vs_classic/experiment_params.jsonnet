local params = import '../model_params.libsonnet';
local bps = 3;
local opt = 'adam';
local signal_power = 1.0;
{
    "num_trials": 50,
    "train_SNR_dbs": [
        18.2,
        13.2,
        8.4
    ],
    "base": {
        "__meta__": {
            "protocol": "private_preamble",
            "experiment_name": "8PSK_neural_vs_classic",
            "mod_order": "8PSK",
            "random_seed": "placeholder",
            "numpy_seed": "placeholder",
            "torch_seed": "placeholder",
            "verbose": false
        },
        "test_batch_size": 100000,
        "test_SNR_db_type": "ber_roundtrip",
        "bits_per_symbol": bps,
        "batch_size": 64,
        "num_iterations": 10000,
        "results_every": 500,
        "signal_power": signal_power,
        "agent1": {
            "bits_per_symbol": bps,
            "max_amplitude": signal_power,
            "optimizer": opt,
            "mod_model": "neural",
            "mod_params": params['neural_mod_8psk'],
            "demod_model": "neural",
            "demod_params": params['neural_demod_8psk']
        },
        "agent2": {
            "bits_per_symbol": bps,
            "max_amplitude": signal_power,
            "optimizer": opt,
            "mod_model": "classic",
            "mod_params": {},
            "demod_model": "classic",
            "demod_params": {}
        }
    }
}