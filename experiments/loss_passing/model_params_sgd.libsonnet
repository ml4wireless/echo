local lambda_center = 0.0;
local lambda_l1 = 0.0;
local lambda_l2 = 0.0;
local restrict_energy = 1.0;

{
    'neural_mod_qpsk':{
        'hidden_layers':            [20],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-1,
                                        'max_val': 1.1e-1,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  }, 
        'initial_std':              3e-1,
        'min_std':                  1e-3,
        'max_std':                  100,
        'lambda_prob':              1e-10,
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            lambda_center,
    },

    'neural_demod_qpsk' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-1,
                                        'max_val': 1.1e-1,  },
        'cross_entropy_weight':     1.0, # iterate over
    },

    'neural_mod_8psk' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 5.0e-2,
                                        'max_val': 5.1e-2,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  },
        'initial_std':              4e-1,
        'min_std':                  1e-3,
        'max_std':                  100,
        'lambda_prob':              1e-10,
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            lambda_center,
    },

    'neural_demod_8psk' : {
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  },
        'cross_entropy_weight':     1.0, # iterate over
    },

    'neural_mod_qam16': {
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-1,
                                        'max_val': 1.1e-1,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  }, 
        'initial_std':              4e-1,
        'min_std':                  1e-3,
        'max_std':                  100,
        'lambda_prob':              1e-10,
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            lambda_center,
    },

    'neural_demod_qam16': {
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-1,
                                        'max_val': 1.1e-1,  },
        'cross_entropy_weight':     1.0, # iterate over
    },

}
