local restrict_energy = 1.0;

{
  'neural_mod_qpsk' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'stepsize_mu':              3e-2,
        'restrict_energy':          restrict_energy,
        'lambda_center':            0.0,
    },
    'neural_demod_qpsk' : {
        'hidden_layers':            [50],
        'activation_fn_hidden':     'tanh',
        'loss_type':                'l2',
        'stepsize_cross_entropy':   3e-2,
        'cross_entropy_weight':     1.0,
    },
    'poly_mod_qpsk' : {
        'stepsize_mu': 0.1,
        'lambda_l1': 0,
        'lambda_p': 0,
        'restrict_energy': restrict_energy,
        'bits_per_symbol': 2,
        'loss_function': 'vanilla_pg',
        'optimizer': 'adam',
        'lambda_center': 0,
    },
    'poly_demod_qpsk': {
        'cross_entropy_weight':     1.0,
        'degree_polynomial': 1,
        'epochs': 1,
        'lambda_l1' : 1e-3,
        'loss_type': 'l2',
        'stepsize_cross_entropy': 1e-1,
        'bits_per_symbol': 2,
        'optimizer': 'adam',
    },
}
