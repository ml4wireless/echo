local lambda_center = 0.0;
local lambda_l1 = 0.0;
local lambda_l2 = 0.0;
local restrict_energy = 1.0;

{
    'poly_mod_qpsk' : {
                "initial_std": 1.0,
                "min_std":0.2,
                "max_std":2.0,
                "lambda_center": 0,
                "lambda_l1": 0.0,
                "lambda_p": 0,
                "restrict_energy": 1,
                "stepsize_mu": 0.04,
                "stepsize_sigma": 4e-3,
                "bits_per_symbol": 2,
                "loss_function": "vanilla_pg",
                "max_amplitude": 1,
                "optimizer": "adam"
},


    'poly_demod_qpsk':{
                "cross_entropy_weight": 1,
                "degree_polynomial": 1,
                "epochs": 1,
                "lambda_l1": 0.001,
                "loss_type": "l2",
                "stepsize_cross_entropy": 1e-2,
                "bits_per_symbol": 2,
                "max_amplitude": 1,
                "optimizer": "adam"
},

    'neural_mod_qpsk' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              8e-3,
        'stepsize_sigma':           1e-4,
        'initial_std':              0.3,
        'min_std':                  0.1,
        'max_std':                  1.0,
        'lambda_prob':              1.0e-10, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            0.0, 
    },
    'neural_demod_qpsk' : {
        'hidden_layers':            [50], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   5e-3,
        'cross_entropy_weight':     1.0, 
    },

    'neural_mod_8psk' : {
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              5e-3,
        'stepsize_sigma':           1e-5,
        'initial_std':              0.2,
        'min_std':                  0.1,
        'max_std':                  1.0,
        'lambda_prob':              1e-10,
        'restrict_energy':          restrict_energy,
        'lambda_center':            lambda_center,
    },

    'neural_demod_8psk' : {
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'loss_type':                'l2',
        'stepsize_cross_entropy':   1e-2,
        'cross_entropy_weight':     1.0,
    },
    
    'neural_mod_qam16':{
        'hidden_layers':           	[200],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              5e-3,
        'stepsize_sigma':           1e-4,
        'initial_std':              0.2,
        'min_std':                  0.1,
        'max_std':                  1.0,
        'lambda_prob':              1.0e-10, 
        'restrict_energy':          restrict_energy,
        'lambda_center':            lambda_center, 
    },

    'neural_demod_qam16' : {
        'hidden_layers':            [200], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   1e-2,
        'cross_entropy_weight':     1.0, # iterate over  
    },

}
