local params = import '../model_params.libsonnet';
local bps = 2;
local opt = 'adam';
local signal_power = 1.0;
{
    "num_trials": 50,
    "train_SNR_dbs": [
        13.0,
        8.4,
        4.2
    ],
    "base": {
        "__meta__": {
            "protocol": "private_preamble",
            "experiment_name": "QPSK_poly_vs_clone",
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
        "num_iterations": 10000,
        "results_every": 500,
        "signal_power": signal_power,
        "agent1": {
            "bits_per_symbol": bps,
            "max_amplitude": signal_power,
            "optimizer": opt,
            "mod_model": "poly",
            "mod_params": params['poly_mod_qpsk'],
            "demod_model": "poly",
            "demod_params": params['poly_demod_qpsk']
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