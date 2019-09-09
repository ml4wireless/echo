failed_test1 = {'bits_per_symbol': 2,
                'optimizer': 'adam',
                'max_amplitude': 1.0,
                'demod_model': 'poly',
                'mod_model': 'poly',
                'mod_params': {'bits_per_symbol': 2,
                               'stepsize_mu': 0.116569688099379,
                               'stepsize_sigma': 1.2657569599745331e-05,
                               'initial_std': 0.5, 'max_std': 100.0, 'min_std': 0.0001,
                               'lambda_center': 0.04614909579851473,
                               'lambda_l1': 0.0, 'lambda_l2': 0.0, 'restrict_energy': 1, 'max_amplitude': 1.0,
                               'optimizer': 'adam'},
                'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                                 'stepsize_cross_entropy': 0.00033916254327282696,
                                 'cross_entropy_weight': 1.0, 'epochs': 5, 'degree_polynomial': 4, 'lambda_l1': 0.0,
                                 'lambda_l2': 0.0}}
failed_test1_BER = 0.2138

test2 = {'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly', 'mod_model': 'poly',
         'mod_params': {'bits_per_symbol': 2,
                        'stepsize_mu': 0.020752168467175626,
                        'stepsize_sigma': 1.4253099726872408e-05, 'initial_std': 0.5, 'max_std': 100.0,
                        'min_std': 0.0001,
                        'lambda_center': 0.041390662788855816, 'lambda_l1': 0.0, 'lambda_l2': 0.0, 'restrict_energy': 1,
                        'max_amplitude': 1.0, 'optimizer': 'adam'},
         'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                          'stepsize_cross_entropy': 0.00031903666641362846,
                          'cross_entropy_weight': 1.0, 'epochs': 5, 'degree_polynomial': 2, 'lambda_l1': 0.0,
                          'lambda_l2': 0.0}}

test2_BER = 0.2188
test2_BER = "24/32 got between 0.2 to 0.3"

failed_test3 = {'test': 3, 'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly',
                'mod_model': 'poly',
                'mod_params': {'bits_per_symbol': 2,
                               'stepsize_mu': 0.44663272907924434,
                               'stepsize_sigma': 0.004310871042042645,
                               'initial_std': 0.5, 'max_std': 100.0, 'min_std': 0.0001,
                               'lambda_center': 0.0848420579843279,
                               'lambda_l1': 0.0, 'lambda_l2': 0.0, 'restrict_energy': 1, 'max_amplitude': 1.0,
                               'optimizer': 'adam'},
                'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                                 'stepsize_cross_entropy': 0.04006124218832799,
                                 'cross_entropy_weight': 1.0, 'epochs': 2, 'degree_polynomial': 3,
                                 'lambda_l1': 0.004778105122132471, 'lambda_l2': 0.0}}
failed_test3_BER = 0.2148

failed_test4 = {'test': 4, 'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly',
                'mod_model': 'poly',
                'mod_params': {'bits_per_symbol': 2, 'stepsize_mu': 0.4381201500079638,
                               'stepsize_sigma': 0.0031354682754022828,
                               'initial_std': 0.5, 'max_std': 100.0, 'min_std': 0.0001,
                               'lambda_center': 0.09060347692216711,
                               'lambda_l1': 0.0, 'lambda_l2': 0.0, 'restrict_energy': 1, 'max_amplitude': 1.0,
                               'optimizer': 'adam'},
                'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                                 'stepsize_cross_entropy': 0.027119727824067522,
                                 'cross_entropy_weight': 1.0, 'epochs': 2, 'degree_polynomial': 3,
                                 'lambda_l1': 0.005603781749916268,
                                 'lambda_l2': 0.0}}
failed_test4_BER = 0.2152

failed_test5 = {'test': 5, 'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly',
                'mod_model': 'poly',
                'mod_params': {'bits_per_symbol': 2, 'stepsize_mu': 0.44663272907924434,
                               'stepsize_sigma': 0.004310871042042645,
                               'initial_std': 0.5, 'max_std': 100.0, 'min_std': 0.0001,
                               'lambda_center': 0.0848420579843279,
                               'lambda_l1': 0.0, 'lambda_l2': 0.0, 'restrict_energy': 1, 'max_amplitude': 1.0,
                               'optimizer': 'adam'},
                'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                                 'stepsize_cross_entropy': 0.04006124218832799,
                                 'cross_entropy_weight': 1.0, 'epochs': 2, 'degree_polynomial': 3,
                                 'lambda_l1': 0.004778105122132471,
                                 'lambda_l2': 0.0}}
failed_test5_BER = 0.215

failed_test6 = {'test': 6, 'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly',
                'mod_model': 'poly',
                'mod_params': {'bits_per_symbol': 2, 'stepsize_mu': 0.4381201500079638,
                               'stepsize_sigma': 0.0031354682754022828,
                               'initial_std': 0.5, 'max_std': 100.0, 'min_std': 0.0001,
                               'lambda_center': 0.09060347692216711,
                               'lambda_l1': 0.0, 'lambda_l2': 0.0, 'restrict_energy': 1, 'max_amplitude': 1.0,
                               'optimizer': 'adam'},
                'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                                 'stepsize_cross_entropy': 0.027119727824067522,
                                 'cross_entropy_weight': 1.0, 'epochs': 2, 'degree_polynomial': 3,
                                 'lambda_l1': 0.005603781749916268, 'lambda_l2': 0.0}}
failed_test6_BER = 0.215

###

test7 = {'test': 7, 'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly',
         'mod_model': 'poly',
         'mod_params': {'bits_per_symbol': 2,
                        'stepsize_mu': 0.020503989577293397,
                        'stepsize_sigma': 1.531678771972656e-05,
                        'initial_std': 0.5, 'max_std': 100.0, 'min_std': 0.0001,
                        'lambda_center': 0.012652252332824576,
                        'lambda_l1': 2.8537917769021694e-05, 'lambda_l2': 0.0, 'restrict_energy': 1,
                        'max_amplitude': 1.0,
                        'optimizer': 'adam'},
         'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                          'stepsize_cross_entropy': 0.0015560648338576175,
                          'cross_entropy_weight': 1.0,
                          'epochs': 2, 'degree_polynomial': 2,
                          'lambda_l1': 0.002524751159050059,
                          'lambda_l2': 0.0}}
test7_BER = 0.1991
test7_result = "26/32 got between 0.2 to 0.3"


####
test8 = {'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly', 'mod_model': 'poly',
         'mod_params': {'bits_per_symbol': 2,
                        'stepsize_mu': 0.020703293295467603,
                        'stepsize_sigma': 1.5612775721048054e-05, 'initial_std': 0.5, 'max_std': 100.0,
                        'min_std': 0.0001,
                        'lambda_center': 0.09023202225265868,
                        'lambda_l1': 2.2196537912363644e-05, 'lambda_l2': 0.0,
                        'restrict_energy': 1, 'max_amplitude': 1.0, 'optimizer': 'adam'},
         'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                          'stepsize_cross_entropy': 0.0012902219671604848,
                          'cross_entropy_weight': 1.0,
                          'epochs': 5, 'degree_polynomial': 2,
                          'lambda_l1': 0.009226154102664275, 'lambda_l2': 0.0}}
test8_BER = 0.2078

test9 = {'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly', 'mod_model': 'poly',
         'mod_params': {'bits_per_symbol': 2, 'stepsize_mu': 0.020868330597877502,
                        'stepsize_sigma': 1.5032864689826966e-05, 'initial_std': 0.5, 'max_std': 100.0,
                        'min_std': 0.0001,
                        'lambda_center': 0.027106437761030444, 'lambda_l1': 2.935572779695818e-05, 'lambda_l2': 0.0,
                        'restrict_energy': 1, 'max_amplitude': 1.0, 'optimizer': 'adam'},
         'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam', 'stepsize_cross_entropy': 0.002406928393022335,
                          'cross_entropy_weight': 1.0, 'epochs': 4, 'degree_polynomial': 2,
                          'lambda_l1': 9.732253662727146e-05, 'lambda_l2': 0.0}}

test9_BER = 0.2046

test10 = {'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0, 'demod_model': 'poly', 'mod_model': 'poly',
          'mod_params': {'bits_per_symbol': 2,
                         'stepsize_mu': 0.020998466330416063,
                         'stepsize_sigma': 1.5400305584857337e-05, 'initial_std': 0.5, 'max_std': 100.0,
                         'min_std': 0.0001,
                         'lambda_center': 0.07531378255234145,
                         'lambda_l1': 5.667579464310951e-05, 'lambda_l2': 0.0,
                         'restrict_energy': 1, 'max_amplitude': 1.0, 'optimizer': 'adam'},
          'demod_params': {'bits_per_symbol': 2, 'optimizer': 'adam',
                           'stepsize_cross_entropy': 0.0014345904872943441,
                           'cross_entropy_weight': 1.0, 'epochs': 3, 'degree_polynomial': 2,
                           'lambda_l1': 2.2769777637064027e-05, 'lambda_l2': 0.0}}
test10_BER = 0.2036
###

WINNER = {'bits_per_symbol': 2, 'optimizer': 'adam', 'max_amplitude': 1.0,
       'mod_model': 'poly',
       'mod_params': {'bits_per_symbol': 2,
                      'stepsize_mu': 2.1e-2, #0.020503989577293397,
                      'stepsize_sigma': 1.53e-5, #1.531678771972656e-05,
                      'initial_std': 0.5, 'max_std': 100.0, 'min_std': 0.0001,
                      'lambda_center': 1.26e-2, #0.012652252332824576,
                      'lambda_l1': 3e-5, #2.8537917769021694e-05,
                      'lambda_l2': 0.0,
                      'restrict_energy': 1,
                      'max_amplitude': 1.0,
                      'optimizer': 'adam'},
       'demod_model': 'poly',
       'demod_params': {'bits_per_symbol': 2,
                        'optimizer': 'adam',
                        'stepsize_cross_entropy': 1.55e-3, #0.0015560648338576175,
                        'cross_entropy_weight': 1.0,
                        'epochs': 2, 'degree_polynomial': 2,
                        'lambda_l1': 2.52e-3, #0.002524751159050059,
                        'lambda_l2': 0.0}
        }