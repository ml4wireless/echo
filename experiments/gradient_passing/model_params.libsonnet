local lambda_center = 0.0;
local lambda_l1 = 0.0;
local lambda_l2 = 0.0;
local restrict_energy = 1.0;

{
    #POLYNOMIAL 
    'poly_mod_qpsk':{
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 8.0e-2,
                                        'max_val': 8.1e-2,  },
        'lambda_l1':                lambda_l1, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    'poly_demod_qpsk':{
        'degree_polynomial' :       2,
        'loss_type' :               'l2',
        'stepsize_cross_entropy' :  {   'sample': 'Uniform' ,
                                        'min_val': 8.0e-2,
                                        'max_val': 8.1e-2,  },
        'lambda_l1':                lambda_l1, 
        'cross_entropy_weight':     1.0, 
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    'poly_mod_8psk':{
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 5.0e-2,
                                        'max_val': 5.1e-2,  },
        'lambda_l1':                lambda_l1, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    'poly_demod_8psk':{
        'degree_polynomial' :       2,
        'loss_type' :               'l2',
        'stepsize_cross_entropy' :  {   'sample': 'Uniform' ,
                                        'min_val': 5.0e-3,
                                        'max_val': 5.1e-3,  },
        'lambda_l1':                lambda_l1, 
        'cross_entropy_weight':     1.0, 
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    'poly_mod_qam16':{
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 5.0e-3,
                                        'max_val': 5.1e-3,  },
        'lambda_l1':                0.00, #.05 (circles) #.06 (grids) #.07 (lines) with polypoly
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    #FOR POLY_CLASSIC
    'polyclassic_mod_qam16':{
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 5.0e-3,
                                        'max_val': 5.1e-3,  },
        'lambda_l1':                lambda_l1, 
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'optimizer':                null,
        'lambda_center':            lambda_center,
    }, 

    'poly_demod_qam16':{
        'degree_polynomial' :       2,
        'loss_type' :               'l2',
        'stepsize_cross_entropy' :  {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  },
        'lambda_l1' :               lambda_l1,
        'cross_entropy_weight':     1.0, 
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    #NEURAL
    'neural_mod_qpsk':{
        'hidden_layers':            [25],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 9.0e-3,
                                        'max_val': 9.1e-3,  },
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    'neural_demod_qpsk':{
        'hidden_layers':            [50], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-2,
                                        'max_val': 1.1e-2,  }, 
        'cross_entropy_weight':     1.0, # iterate over  
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    'neural_mod_8psk':{
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  },
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    'neural_demod_8psk':{
        'hidden_layers':            [100], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  }, 
        'cross_entropy_weight':     1.0, # iterate over  
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },
    
    'neural_mod_qam16':{
        'hidden_layers':            [100],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  },
        'restrict_energy':          restrict_energy,
        'lambda_p':                 0.0,
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

    'neural_demod_qam16':{
        'hidden_layers':            [100], 
        'activation_fn_hidden':     'tanh',   
        'loss_type':                'l2',
        'stepsize_cross_entropy':   {   'sample': 'Uniform' ,
                                        'min_val': 1.0e-3,
                                        'max_val': 1.1e-3,  }, 
        'cross_entropy_weight':     1.0, # iterate over  
        'optimizer':                null,
        'lambda_center':            lambda_center,
    },

}