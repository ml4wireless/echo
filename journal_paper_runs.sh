#!/bin/bash

### This script is meant to be run on the BRC
### See scripts/single for an example of running locally.

### Plot 1 ###
# QPSK neurals & polys vs classics EPP
cd experiments/
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 classic --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_classic --num_iterations 200 --num_results_logged 40
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 poly --mod2 classic --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_fast_vs_classic --num_iterations 500 --num_results_logged 50

python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 classic --model_params_template $(pwd)/../custom_model_params/model_params_slow.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_slow_vs_classic --num_iterations 500 --num_results_logged 50
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 poly --mod2 classic --model_params_template $(pwd)/../custom_model_params/model_params_slow.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_slow_vs_classic --num_iterations 500 --num_results_logged 50

python make_jobs.py --experiment_folder private_preamble/QPSK_neural_fast_vs_classic
python make_jobs.py --experiment_folder private_preamble/QPSK_poly_fast_vs_classic
python make_jobs.py --experiment_folder private_preamble/QPSK_neural_slow_vs_classic
python make_jobs.py --experiment_folder private_preamble/QPSK_poly_slow_vs_classic

cd ..

./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_fast_vs_classic
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_slow_vs_classic
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_poly_fast_vs_classic
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_poly_slow_vs_classic

### Plot 2 ###
# QPSK EPP neural ff,fs,ss and poly ff,fs,ss
cd experiments/
# Neural
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 neural --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_neural_fast --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --num_results_logged 30 --num_iterations 500
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 neural --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_slow_vs_neural_slow --model_params_template $(pwd)/../custom_model_params/model_params_slow.libsonnet --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 neural --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_neural_slow --mod1_param_json $(pwd)/../custom_model_params/mod_neural_fast_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_neural_fast_params.json --mod2_param_json $(pwd)/../custom_model_params/mod_neural_slow_params.json --demod2_param_json $(pwd)/../custom_model_params/demod_neural_slow_params.json --num_results_logged 30
# Poly
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 poly --mod2 poly --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_fast_vs_poly_fast --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 poly --mod2 poly --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_slow_vs_poly_slow --model_params_template $(pwd)/../custom_model_params/model_params_slow.libsonnet --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 poly --mod2 poly --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_fast_vs_poly_slow --mod1_param_json $(pwd)/../custom_model_params/mod_poly_fast_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_poly_fast_params.json --mod2_param_json $(pwd)/../custom_model_params/mod_poly_slow_params.json --demod2_param_json $(pwd)/../custom_model_params/demod_poly_slow_params.json --num_results_logged 30
# Jobs
python make_jobs.py --experiment_folder private_preamble/QPSK_neural_fast_vs_neural_fast
python make_jobs.py --experiment_folder private_preamble/QPSK_neural_slow_vs_neural_slow
python make_jobs.py --experiment_folder private_preamble/QPSK_neural_fast_vs_neural_slow
python make_jobs.py --experiment_folder private_preamble/QPSK_poly_fast_vs_poly_fast
python make_jobs.py --experiment_folder private_preamble/QPSK_poly_slow_vs_poly_slow
python make_jobs.py --experiment_folder private_preamble/QPSK_poly_fast_vs_poly_slow

cd ..

./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_fast_vs_neural_fast
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_slow_vs_neural_slow
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_fast_vs_neural_slow
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_poly_fast_vs_poly_fast
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_poly_slow_vs_poly_slow
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_poly_fast_vs_poly_slow

### Plot 3 ###
# QPSK EPP poly ss and neural ff and poly s vs neural f
cd experiments/
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 poly --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_poly_slow --mod1_param_json $(pwd)/../custom_model_params/mod_neural_fast_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_neural_fast_params.json --mod2_param_json $(pwd)/../custom_model_params/mod_poly_slow_params.json --demod2_param_json $(pwd)/../custom_model_params/demod_poly_slow_params.json --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 poly --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_slow_vs_poly_slow --mod1_param_json $(pwd)/../custom_model_params/mod_neural_slow_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_neural_slow_params.json --mod2_param_json $(pwd)/../custom_model_params/mod_poly_slow_params.json --demod2_param_json $(pwd)/../custom_model_params/demod_poly_slow_params.json --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 poly --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_poly_fast --mod1_param_json $(pwd)/../custom_model_params/mod_neural_fast_params.json --demod1_param_json $(pwd)/../custom_model_params/demod_neural_fast_params.json --mod2_param_json $(pwd)/../custom_model_params/mod_poly_fast_params.json --demod2_param_json $(pwd)/../custom_model_params/demod_poly_fast_params.json --num_results_logged 30

python make_jobs.py --experiment_folder private_preamble/QPSK_neural_fast_vs_poly_slow
python make_jobs.py --experiment_folder private_preamble/QPSK_neural_slow_vs_poly_slow
python make_jobs.py --experiment_folder private_preamble/QPSK_neural_fast_vs_poly_fast

cd ..

./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_fast_vs_poly_slow
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_slow_vs_poly_slow
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_fast_vs_poly_fast

### Plot 5 ###
# QPSK GP LP ESP EPP neural f vs classic
cd experiments/
python create_experiment_params.py --mod_order qpsk --protocol shared_preamble --mod1 neural --mod2 classic --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_classic --num_iterations 300 --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol loss_passing --mod1 neural --demod1 classic  --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_classic --num_results_logged 30 --num_iterations 200
python create_experiment_params.py --mod_order qpsk --protocol gradient_passing --mod1 neural --demod1 classic --model_params_template $(pwd)/../custom_model_params/model_params_gp.libsonnet  --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_gp_vs_classic --num_results_logged 30 --num_iterations 200

python make_jobs.py --experiment_folder shared_preamble/QPSK_neural_fast_vs_classic
python make_jobs.py --experiment_folder loss_passing/QPSK_neural_fast_vs_classic
python make_jobs.py --experiment_folder gradient_passing/QPSK_neural_gp_vs_classic

cd ..

./runecho_args -b -n 1 -j experiments/shared_preamble/QPSK_neural_fast_vs_classic
./runecho_args -b -n 1 -j experiments/loss_passing/QPSK_neural_fast_vs_classic
./runecho_args -b -n 1 -j experiments/gradient_passing/QPSK_neural_gp_vs_classic

### Plot 6 ###
# QPSK GP LP ESP EPP poly f vs classic
cd experiments/
python create_experiment_params.py --mod_order qpsk --protocol shared_preamble --mod1 poly --mod2 classic --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_fast_vs_classic --num_iterations 300 --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol loss_passing --mod1 poly --demod1 classic --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_fast_vs_classic --num_results_logged 30 --num_iterations 300
python create_experiment_params.py --mod_order qpsk --protocol gradient_passing --mod1 poly --demod1 classic --model_params_template $(pwd)/../custom_model_params/model_params_gp.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_gp_vs_classic --num_results_logged 30 --num_iterations 300

python make_jobs.py --experiment_folder shared_preamble/QPSK_poly_fast_vs_classic
python make_jobs.py --experiment_folder loss_passing/QPSK_poly_fast_vs_classic
python make_jobs.py --experiment_folder gradient_passing/QPSK_poly_gp_vs_classic

cd ..

./runecho_args -b -n 1 -j experiments/shared_preamble/QPSK_poly_fast_vs_classic
./runecho_args -b -n 1 -j experiments/loss_passing/QPSK_poly_fast_vs_classic
./runecho_args -b -n 1 -j experiments/gradient_passing/QPSK_poly_gp_vs_classic

### Plot 7 ###
# QPSK GP LP ESP EPP neural f vs neural f
cd experiments/
python create_experiment_params.py --mod_order qpsk --protocol shared_preamble --mod1 neural --mod2 neural --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_neural_fast --num_iterations 500 --num_results_logged 60
python create_experiment_params.py --mod_order qpsk --protocol loss_passing --mod1 neural --demod1 neural --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_neural_fast --num_results_logged 60 --num_iterations 300
python create_experiment_params.py --mod_order qpsk --protocol gradient_passing --mod1 neural --demod1 neural --model_params_template $(pwd)/../custom_model_params/model_params_gp.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_gp_vs_neural_gp --num_results_logged 60 --num_iterations 300

python make_jobs.py --experiment_folder shared_preamble/QPSK_neural_fast_vs_neural_fast
python make_jobs.py --experiment_folder loss_passing/QPSK_neural_fast_vs_neural_fast
python make_jobs.py --experiment_folder gradient_passing/QPSK_neural_gp_vs_neural_gp

cd ..

./runecho_args -b -n 1 -j experiments/shared_preamble/QPSK_neural_fast_vs_neural_fast
./runecho_args -b -n 1 -j experiments/loss_passing/QPSK_neural_fast_vs_neural_fast
./runecho_args -b -n 1 -j experiments/gradient_passing/QPSK_neural_gp_vs_neural_gp

### Plot 8 ###
# QPSK GP LP ESP EPP poly f vs poly f
cd experiments/
python create_experiment_params.py --mod_order qpsk --protocol shared_preamble --mod1 poly --mod2 poly --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_fast_vs_poly_fast --num_iterations 2500 --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol loss_passing --mod1 poly --demod1 poly --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_fast_vs_poly_fast --num_results_logged 30 --num_iterations 300
python create_experiment_params.py --mod_order qpsk --protocol gradient_passing --mod1 poly --demod1 poly --model_params_template $(pwd)/../custom_model_params/model_params_gp.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_poly_gp_vs_poly_gp --num_results_logged 30 --num_iterations 300

python make_jobs.py --experiment_folder shared_preamble/QPSK_poly_fast_vs_poly_fast
python make_jobs.py --experiment_folder loss_passing/QPSK_poly_fast_vs_poly_fast
python make_jobs.py --experiment_folder gradient_passing/QPSK_poly_gp_vs_poly_gp

cd ..

./runecho_args -b -n 1 -j experiments/shared_preamble/QPSK_poly_fast_vs_poly_fast
./runecho_args -b -n 1 -j experiments/loss_passing/QPSK_poly_fast_vs_poly_fast
./runecho_args -b -n 1 -j experiments/gradient_passing/QPSK_poly_gp_vs_poly_gp

### Plot 9 ###
# QPSK ESP neural f vs classic, neural ff, poly ff, neural f vs poly f
cd experiments/
python create_experiment_params.py --mod_order qpsk --protocol shared_preamble --mod1 neural --mod2 poly --model_params_template $(pwd)/../custom_model_params/model_params_fast.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_fast_vs_poly_fast --num_iterations 2500 --num_results_logged 30

python make_jobs.py --experiment_folder shared_preamble/QPSK_neural_fast_vs_poly_fast

cd ..

./runecho_args -b -n 1 -j experiments/shared_preamble/QPSK_neural_fast_vs_poly_fast

### Plot 10 ###
# ESP EPP neural ff QPSK 8PSK 16QAM
cd experiments/
python create_experiment_params.py --mod_order 8psk --protocol shared_preamble --mod1 neural --mod2 neural --model_params_template $(pwd)/../custom_model_params/higher_order_model_params.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name 8PSK_neural_fast_vs_neural_fast --num_results_logged 30
python create_experiment_params.py --mod_order qam16 --protocol shared_preamble --mod1 neural --mod2 neural --model_params_template $(pwd)/../custom_model_params/higher_order_model_params.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QAM16_neural_fast_vs_neural_fast --num_results_logged 30
python create_experiment_params.py --mod_order 8psk --protocol private_preamble --mod1 neural --mod2 neural --model_params_template $(pwd)/../custom_model_params/higher_order_model_params.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name 8PSK_neural_fast_vs_neural_fast --num_results_logged 30
python create_experiment_params.py --mod_order qam16 --protocol private_preamble --mod1 neural --mod2 neural --model_params_template $(pwd)/../custom_model_params/higher_order_model_params.libsonnet --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QAM16_neural_fast_vs_neural_fast --num_results_logged 30

python make_jobs.py --experiment_folder shared_preamble/8PSK_neural_fast_vs_neural_fast
python make_jobs.py --experiment_folder shared_preamble/QAM16_neural_fast_vs_neural_fast
python make_jobs.py --experiment_folder private_preamble/8PSK_neural_fast_vs_neural_fast
python make_jobs.py --experiment_folder private_preamble/QAM16_neural_fast_vs_neural_fast

cd ..

./runecho_args -b -n 2 -j experiments/shared_preamble/8PSK_neural_fast_vs_neural_fast
./runecho_args -b -n 2 -j experiments/shared_preamble/QAM16_neural_fast_vs_neural_fast
./runecho_args -b -n 2 -j experiments/private_preamble/8PSK_neural_fast_vs_neural_fast
./runecho_args -b -n 2 -j experiments/private_preamble/QAM16_neural_fast_vs_neural_fast

### Plot 11 ###
# QPSK neural ff train SNR low,mid,high
cd experiments/
# Neural
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 neural --train_snr_db low --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_vs_neural_low_snr --model_params_template $(pwd)/../custom_model_params/snr_model_params.libsonnet --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 neural --train_snr_db mid --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_vs_neural_mid_snr --model_params_template $(pwd)/../custom_model_params/snr_model_params.libsonnet --num_results_logged 30
python create_experiment_params.py --mod_order qpsk --protocol private_preamble --mod1 neural --mod2 neural --train_snr_db high --num_trials 50 --batch_size 256 --experiment_name QPSK_neural_vs_neural_high_snr --model_params_template $(pwd)/../custom_model_params/snr_model_params.libsonnet --num_results_logged 30

python make_jobs.py --experiment_folder private_preamble/QPSK_neural_vs_neural_low_snr
python make_jobs.py --experiment_folder private_preamble/QPSK_neural_vs_neural_mid_snr
python make_jobs.py --experiment_folder private_preamble/QPSK_neural_vs_neural_high_snr

cd ..

./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_vs_neural_low_snr
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_vs_neural_mid_snr
./runecho_args -b -n 1 -j experiments/private_preamble/QPSK_neural_vs_neural_high_snr



