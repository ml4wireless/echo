local lambda_center = 0.0;
local lambda_l1 = 0.0;
local lambda_l2 = 0.0;
local restrict_energy = 1.0;

{
     'neural_mod_8psk' : {
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              8e-3,
        'stepsize_sigma':           4e-3,
        'initial_std':              0.2,
        'min_std':                  0.01,
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
        'stepsize_mu':              7e-4,
        'stepsize_sigma':           5e-4,
        'initial_std':              0.1,
        'min_std':                  0.01,
        'max_std':                  1.0,
        'lambda_prob':              1.0e-10,
        'restrict_energy':          restrict_energy,
        'lambda_center':            lambda_center,
    },

    'neural_demod_qam16' : {
        'hidden_layers':            [200],
        'activation_fn_hidden':     'tanh',
        'loss_type':                'l2',
        'stepsize_cross_entropy':   1e-3,
        'cross_entropy_weight':     1.0, # iterate over  
    }
}
