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
            "protocol": "shared_preamble",
            "experiment_name": "8PSK_poly_vs_clone",
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
        "num_iterations": 6000,
        "results_every": 300,
        "signal_power": signal_power,
        "agent1": {
            "bits_per_symbol": bps,
            "max_amplitude": signal_power,
            "optimizer": opt,
            "mod_model": "poly",
            "mod_params": params['poly_mod_8psk'],
            "demod_model": "poly",
            "demod_params": params['poly_demod_8psk']
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