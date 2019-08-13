local lambda_center = 0.0;
local lambda_l1 = 0.0;
local lambda_l2 = 0.0;
local restrict_energy = 1.0;

{
    'poly_mod_qpsk' : {
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 2.0e-2,
                                        'max_val': 2.1e-2,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1.5e-3,
                                        'max_val': 1.6e-3,  },
        'initial_std':              0.3,
        'min_std':                  0.0001,
        'max_std':                  100,
        'lambda_l1':                {   'sample': 'Uniform' ,
                                        'min_val': 3e-5,
                                        'max_val': 3.1e-5,  },
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            {   'sample': 'Uniform' ,
                                        'min_val': 1.2e-2,
                                        'max_val': 1.3e-2,  }
},


    'poly_demod_qpsk':{
        'degree_polynomial' :       2,
        'epochs':                   2,
        'loss_type' :               'l2',
        'stepsize_cross_entropy' :  {   'sample': 'Uniform' ,
                                        'min_val': 1.5e-3,
                                        'max_val': 1.6e-3,  },
        'lambda_l1':                {   'sample': 'Uniform' ,
                                        'min_val': 2.5e-3,
                                        'max_val': 2.6e-3,  },
        'cross_entropy_weight':     1.0, 
    },

    'poly_mod_8psk':{
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-4,
                                        'max_val': 1.1e-4,  },
        'initial_std':              4.0e-1,
        'min_std':                  1.0e-3,
        'max_std':                  100,
        'lambda_l1':                lambda_l1, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            lambda_center,
    },

    'poly_demod_8psk':{
        'degree_polynomial' :       2,
        'loss_type' :               'l2',
        'stepsize_cross_entropy' :  {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  },
        'lambda_l1':                lambda_l1, 
        'cross_entropy_weight':     1.0, 
    },

    'poly_mod_qam16':{
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  },
        'initial_std':              5.0e-1,
        'min_std':                  1.0e-3,
        'max_std':                  100,
        'lambda_l1':                lambda_l1, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            lambda_center,

    },

    'poly_demod_qam16':{
        'degree_polynomial' :       3,
        'loss_type' :               'l2',
        'stepsize_cross_entropy' :  {   'sample': 'Uniform' ,
                                        'min_val': 8.0e-2,
                                        'max_val': 8.1e-2,  },
        'lambda_l1' :               lambda_l1,
        'cross_entropy_weight':     1.0, 
    },

    'neural_mod_qpsk' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 2.0e-4,
                                        'max_val': 2.1e-4,  }, 
        'initial_std':              2.0e-1,
        'min_std':                  1.0e-3,
        'max_std':                  100,
        'lambda_prob':              1.0e-10, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            lambda_center, 
    },


    'neural_mod_qpsk_replicate_gnuradio' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              1.0e-3,
        'stepsize_sigma':           1.0e-4, 
        'initial_std':              2.0e-1,
        'min_std':                  1.0e-3,
        'max_std':                  100,
        'lambda_prob':              1.0e-10, 
        'restrict_energy':          1.0,
        'max_amplitude':            0.5,
        'lambda_p':                 0.0,
        'lambda_center':            125.0, 
    },


    'neural_demod_qpsk' : {
        'hidden_layers':            [50], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  },
        'cross_entropy_weight':     1.0, 
    },
    


    'neural_demod_qpsk_replicate_gnuradio' : {
        'hidden_layers':            [50], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   1.0e-2, 
        'cross_entropy_weight':     1.0,
    },

    'neural_mod_8psk' : {
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-4,
                                        'max_val': 1.1e-4,  }, 
        'initial_std':              2.0e-1,
        'min_std':                  1.0e-3,
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
        'cross_entropy_weight':     1.0, 
    },

    'neural_mod_qam16':{
        'hidden_layers':           	[100],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-4,
                                        'max_val': 1.1e-4,  }, 
        'initial_std':              1.0e-1,
        'min_std':                  1.0e-3,
        'max_std':                  100,
        'lambda_prob':              1.0e-10, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            lambda_center, 
    },

    'neural_demod_qam16' : {
        'hidden_layers':            [200], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  }, 
        'cross_entropy_weight':     1.0, # iterate over  
    }
}