local lambda_center = 0.0;
local lambda_l1 = 0.0;
local lambda_l2 = 0.0;
local restrict_energy = 1.0;

{
     'neural_mod_qpsk' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              1e-3,
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
        'stepsize_cross_entropy':   1e-3,
        'cross_entropy_weight':     1.0,
    }
}
