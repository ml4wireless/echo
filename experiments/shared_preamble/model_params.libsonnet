local lambda_center = 0.0;
local lambda_l1 = 0.0;
local lambda_l2 = 0.0;
local restrict_energy = 1.0;

{
    'poly_mod_qpsk' : {
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val':1e-2,
                                        'max_val': 1e-2,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1e-3,
                                        'max_val': 1e-3,  },
        'initial_std':              1,
        'min_std':                  0.1,
        'max_std':                  100,
        'lambda_l1':                {   'sample': 'Uniform' ,
                                        'min_val': 0,
                                        'max_val': 0,  },
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            0.0,
},


    'poly_demod_qpsk':{
        'degree_polynomial' :       1,
        'epochs':                   1,
        'loss_type' :               'l2',
        'stepsize_cross_entropy' :  {   'sample': 'Uniform' ,
                                        'min_val': 1e-3,
                                        'max_val': 1e-3,  },
        'lambda_l1':                {   'sample': 'Uniform' ,
                                        'min_val': 1e-3,
                                        'max_val': 1e-3,  },
        'cross_entropy_weight':     1.0, 
    },


    'neural_mod_qpsk' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.0e-3,  },
        'stepsize_sigma':           {   'sample': 'Uniform' ,
                                        'min_val': 1e-4,
                                        'max_val': 1e-4,  }, 
        'initial_std':              0.2,
        'min_std':                  5.0e-2,
        'max_std':                  100,
        'lambda_prob':              1.0e-10, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'lambda_center':            0.0, 
    },


    'neural_demod_qpsk' : {
        'hidden_layers':            [50], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.0e-3,  },
        'cross_entropy_weight':     1.0, 
    },
    


}
