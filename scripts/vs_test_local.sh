cd experiments


#Neural fast
# python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 neural \
# --train_snr_db mid --num_trials 5 --batch_size 256 --experiment_name QPSK_neural_fast_vs_neural_fast \
#  --mod1_param_json $(pwd)/../custom_model_params/mod_neural_fast_params.json --demod1_param_json \
#  $(pwd)/../custom_model_params/demod_neural_fast_params.json --mod2_param_json $(pwd)/../custom_model_params/mod_neural_fast_params.json \
#   --demod2_param_json $(pwd)/../custom_model_params/demod_neural_fast_params.json --num_results_logged 10 --num_iterations 500

# python make_jobs.py --experiment_folder private_preamble/QPSK_neural_fast_vs_neural_fast

# cd ..
# ./runecho_args -a  -j ./experiments/private_preamble/QPSK_neural_fast_vs_neural_fast


#Neural slow
# cd experiments

# python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 neural \
# --train_snr_db mid --num_trials 5 --batch_size 256 --experiment_name QPSK_neural_slow_vs_neural_slow \
#  --mod1_param_json $(pwd)/../custom_model_params/mod_neural_slow_params.json --demod1_param_json \
#  $(pwd)/../custom_model_params/demod_neural_slow_params.json --mod2_param_json $(pwd)/../custom_model_params/mod_neural_slow_params.json \
#   --demod2_param_json $(pwd)/../custom_model_params/demod_neural_slow_params.json --num_results_logged 10 --num_iterations 1000

# python make_jobs.py --experiment_folder private_preamble/QPSK_neural_slow_vs_neural_slow 

# cd ..
# ./runecho_args -a  -j ./experiments/private_preamble/QPSK_neural_slow_vs_neural_slow

#Nerual slow vs Neural Fast

# python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 neural  --mod2 neural --train_snr_db mid --num_trials 5 --batch_size 256 --experiment_name QPSK_neural_fast_vs_neural_slow  --mod1_param_json $(pwd)/../custom_model_params/vs_nf_mod.json --demod1_param_json $(pwd)/../custom_model_params/vs_nf_demod.json   --mod2_param_json $(pwd)/../custom_model_params/vs_ns_mod.json   --demod2_param_json $(pwd)/../custom_model_params/vs_ns_demod.json --num_results_logged 10 --num_iterations 1000
# cd experiments


# python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 neural  --mod2 neural --train_snr_db mid --num_trials 5 --batch_size 256 --experiment_name QPSK_neural_fast_vs_neural_slow  --mod1_param_json $(pwd)/../custom_model_params/mod_neural_fast_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_neural_fast_params.json   --mod2_param_json $(pwd)/../custom_model_params/mod_neural_fast_params.json   --demod2_param_json $(pwd)/../custom_model_params/demod_neural_fast_params.json --num_results_logged 10 --num_iterations 1000


# python make_jobs.py --experiment_folder private_preamble/QPSK_neural_fast_vs_neural_slow

# cd ..

# ./runecho_args -a  -j ./experiments/private_preamble/QPSK_neural_fast_vs_neural_slow

# Nerual slow vs Poly Fast

# python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 neural  --mod2 poly --train_snr_db mid --num_trials 5 --batch_size 256 --experiment_name QPSK_neural_slow_vs_poly_fast  --mod1_param_json $(pwd)/../custom_model_params/mod_neural_slow_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_neural_slow_params.json   --mod2_param_json $(pwd)/../custom_model_params/mod_poly_fast_params.json   --demod2_param_json $(pwd)/../custom_model_params/demod_poly_fast_params.json --num_results_logged 10 --num_iterations 5000


# python make_jobs.py --experiment_folder private_preamble/QPSK_neural_slow_vs_poly_fast

# cd ..

# ./runecho_args -a  -j ./experiments/private_preamble/QPSK_neural_slow_vs_poly_fast


#Poly fast vs poly fast
python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 poly  --mod2 poly --train_snr_db mid --num_trials 5 --batch_size 256 --experiment_name QPSK_poly_fast_vs_poly_fast  --mod1_param_json $(pwd)/../custom_model_params/mod_poly_fast_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_poly_fast_params.json   --mod2_param_json $(pwd)/../custom_model_params/mod_poly_fast_params.json   --demod2_param_json $(pwd)/../custom_model_params/demod_poly_fast_params.json --num_results_logged 10 --num_iterations 1500


python make_jobs.py --experiment_folder private_preamble/QPSK_poly_fast_vs_poly_fast

cd ..

./runecho_args -a  -j ./experiments/private_preamble/QPSK_poly_fast_vs_poly_fast



#Poly slow vs poly slow


# python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 poly  --mod2 poly --train_snr_db mid --num_trials 10 --batch_size 256 --experiment_name QPSK_poly_slow_vs_poly_slow  --mod1_param_json $(pwd)/../custom_model_params/mod_poly_slow_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_poly_slow_params.json   --mod2_param_json $(pwd)/../custom_model_params/mod_poly_slow_params.json   --demod2_param_json $(pwd)/../custom_model_params/demod_poly_slow_params.json --num_results_logged 10 --num_iterations 5000


# python make_jobs.py --experiment_folder private_preamble/QPSK_poly_slow_vs_poly_slow

# cd ..

# ./runecho_args -a  -j ./experiments/private_preamble/QPSK_poly_slow_vs_poly_slow


#Poly fast vs poly slow

# python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 poly  --mod2 poly --train_snr_db mid --num_trials 10 --batch_size 256 --experiment_name QPSK_poly_fast_vs_poly_slow  --mod1_param_json $(pwd)/../custom_model_params/mod_poly_fast_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_poly_fast_params.json   --mod2_param_json $(pwd)/../custom_model_params/mod_poly_slow_params.json   --demod2_param_json $(pwd)/../custom_model_params/demod_poly_slow_params.json --num_results_logged 10 --num_iterations 5000


# python make_jobs.py --experiment_folder private_preamble/QPSK_poly_fast_vs_poly_slow

# cd ..

# ./runecho_args -a  -j ./experiments/private_preamble/QPSK_poly_fast_vs_poly_slow



#Neural fast vs poly slow
# python create_experiment_params.py --verbose --mod_order qpsk --protocol private_preamble --mod1 neural  --mod2 poly --train_snr_db mid --num_trials 5 --batch_size 256 --experiment_name QPSK_neural_fast_vs_poly_slow --mod1_param_json $(pwd)/../custom_model_params/mod_neural_fast_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_neural_fast_params.json   --mod2_param_json $(pwd)/../custom_model_params/mod_poly_slow_params.json   --demod2_param_json $(pwd)/../custom_model_params/demod_poly_slow_params.json --num_results_logged 10 --num_iterations 5000


# python make_jobs.py --experiment_folder private_preamble/QPSK_neural_fast_vs_poly_slow

# cd ..

# ./runecho_args -a  -j ./experiments/private_preamble/QPSK_neural_fast_vs_poly_slow
