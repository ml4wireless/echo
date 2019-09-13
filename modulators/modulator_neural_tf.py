import numpy as np
import os
import tensorflow as tf
from utils.util_data import get_bit_l1_loss, integers_to_symbols
from utils.util_tf import normc_initializer
from utils.mod_demod_abstract import Modulator

class ModulatorNeural(Modulator):
    def __init__(self, 
                 seed=7,
                 hidden_layers = [40],
                 bits_per_symbol = 2, 
                 lambda_p=1, 
                 max_std = 1e1,
                 min_std = 1e-5,
                 initial_std=1e-2,
                 restrict_energy = False,
                 activation_fn_hidden = tf.nn.relu,
                 kernel_initializer_hidden = normc_initializer(1.0),
                 bias_initializer_hidden =  tf.glorot_uniform_initializer(),
                 activation_fn_output = None,
                 kernel_initializer_output = normc_initializer(1.0),
                 bias_initializer_output =  tf.glorot_uniform_initializer(),
                 optimizer = tf.train.AdamOptimizer,
                 lambda_prob = 0.1,
                 stepsize_mu=1e-3,
                 stepsize_sigma=1e-5, 
                 **kwargs
                ):           
        '''
        Define neural net parameters, loss, optimizer
        Inputs: 
        seed: the tf.random seed to be used
        hidden_layers: np array of shape [m]/ list of length [m] containining number of units in each hidden layer
        bits_per_symbol: NN takes in input of this size
        lambda_p: Scaling factor for power loss term (used only when restrict_energy is False)
        restrict_energy: If true normalize average outputs(re + 1j*im) to have average energy 1  
        min_std: Minimum standard deviation while exploring 
        initial_std: Initial log standard deviation of exploring
        max_std: Maximum standard deviation while exploring
        activation_fn_hidden: Activation function to be used for hidden layers (default = tf.nn.relu)
        kernel_initializer_hidden:  Kernel intitializer for hidden layers (default = normc_initializer(1.0)) 
        bias_initializer_hidden: Bias initialize for hidden layers (default = tf.glorot_uniform_initializer())
        activation_fn_output: Activation function to be used for output layer (default = None)
        kernel_initializer_output: Kernel intitializer for output layer (default = normc_initializer(1.0))
        bias_initializer_output: Bias initializer for output layer (default = tf.glorot_uniform_initializer())
        optimizer: Optimizer to be used while training (default = tf.train.AdamOptimizer),
        '''

        if activation_fn_hidden == 'relu':
            activation_fn_hidden = tf.nn.relu
        elif activation_fn_hidden == 'tanh':
            activation_fn_hidden = tf.nn.tanh
        elif activation_fn_hidden == 'sigmoid':
            activation_fn_hidden = tf.nn.sigmoid

        normc_std = kwargs.get('kernel_initializer', {}).get('normc_std', False)
        normc_seed = kwargs.get('kernel_initializer', {}).get('normc_seed', False)
        if normc_std and normc_seed:
            init_func = normc_initializer(std=normc_std, seed=normc_seed)
            kernel_initializer_hidden = init_func
            kernel_initializer_output = init_func

        self.times_updated = 0
        self.decoder_updates = 0
        self.mod_class = 'neural'
        self.current_stepsize_mu = stepsize_mu
        self.current_stepsize_sigma = stepsize_sigma
        #Define graph
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Network variables
            tf.set_random_seed(seed)
            self.restrict_energy = restrict_energy
            self.lambda_p = lambda_p
            self.bits_per_symbol = bits_per_symbol

            # Placeholders for training
            self.input = tf.placeholder(tf.float32, [None, self.bits_per_symbol]) # 0 or 1 
            self.echo_labels = tf.placeholder(tf.int32, [None])
            self.constellation_centers = tf.placeholder(tf.float32, [2**self.bits_per_symbol, 2])
            self.echo_probabilities = tf.placeholder(tf.float32, [None, 2**self.bits_per_symbol])
            self.actions_re = tf.placeholder(tf.float32, [None]) 
            self.actions_im = tf.placeholder(tf.float32, [None])           
            self.rewards = tf.placeholder(tf.float32, [None]) # rewardsantages for gradient computation
            self.stepsize_mu = tf.placeholder(shape=[], dtype=tf.float32) 
            self.stepsize_sigma = tf.placeholder(shape=[], dtype=tf.float32) 



            ###############
            # Hidden layers
            ###############
            global_layer_num = 0
            net = self.input
            for cur_layer_num in range(len(hidden_layers)):
                cur_layer_name = 'mu/layer' + str(global_layer_num)        
                global_layer_num +=1
                net = tf.layers.dense(
                        inputs = net,
                        units = hidden_layers[cur_layer_num],
                        activation = activation_fn_hidden, 
                        kernel_initializer = kernel_initializer_hidden,
                        bias_initializer = bias_initializer_hidden,
                        name = cur_layer_name
                        )

            #Define output layer
            cur_layer_name = 'mu/layer' + str(global_layer_num)        
            output = tf.layers.dense(
                          inputs = net,
                          units = 2,
                          activation = activation_fn_output,
                          kernel_initializer = kernel_initializer_output,
                          bias_initializer = bias_initializer_output,
                          name = cur_layer_name
                          )    

            #Means of the constellation point (all the entries corresponding to same input will have same mean)
            self.re_mean = output[:,0]
            self.im_mean = output[:,1]

            ###################
            # Normalize outputs
            ################### 
            if (self.restrict_energy):
                self.max_amplitude = tf.sqrt(tf.reduce_mean(self.re_mean**2 + self.im_mean**2))
                self.normalization = tf.nn.relu(self.max_amplitude-1)+1.0  
                self.re_mean /= self.normalization
                self.im_mean /= self.normalization 

            ####################
            # Floating variables
            ####################
            self.initial_std = tf.cast(initial_std, tf.float32)
            self.re_std = tf.Variable(self.initial_std)
            self.im_std = tf.Variable(self.initial_std)
            sigma_vars = [self.re_std, self.im_std]
            self.sigma_vars = sigma_vars

            ############################
            # Define random distribution
            # and sampled actions
            self.min_std = tf.cast(min_std,tf.float32)
            self.max_std = tf.cast(max_std, tf.float32)
            
            self.re_std_eff = tf.maximum(self.min_std, self.re_std)
            self.im_std_eff = tf.maximum(self.min_std, self.im_std)
        
            self.re_std_eff = tf.minimum(self.max_std, self.re_std_eff)
            self.im_std_eff = tf.minimum(self.max_std, self.im_std_eff)
            
            self.re_distr = tf.distributions.Normal(self.re_mean, self.re_std_eff)
            self.im_distr = tf.distributions.Normal(self.im_mean, self.im_std_eff)

            self.re_sample = self.re_distr.sample()
            self.im_sample = self.im_distr.sample()       

            #######################################
            # Log-probabilities for grad estimation
            #######################################
            self.re_logprob = tf.log(lambda_prob +self.re_distr.prob(self.actions_re))
            self.im_logprob = tf.log(lambda_prob+ self.im_distr.prob(self.actions_im))

            ###############################################
            # Define surrogate loss and optimization tensor
            ###############################################
            self.surr = - tf.reduce_mean(self.rewards * (self.re_logprob + self.im_logprob))
            self.optimizer_mu = optimizer(self.stepsize_mu)
            self.optimizer_sigma = optimizer(self.stepsize_sigma)

            mu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'mu/')
            self.mu_vars = mu_vars
    
            self.update_op_mu= self.optimizer_mu.minimize(self.surr, var_list=mu_vars)
            self.update_op_sigma= self.optimizer_sigma.minimize(self.surr, var_list=sigma_vars)

            ###############
            # Start session
            ###############
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
            

    def modulate(self, data_si, mode='explore', **kwargs):
        '''
        Takes in bit stream and modulates it to complex numbers based on current NN weights and biases
        Inputs:
        data_si: np.array of type integer of shape [N] containing integer representation of bits
        Output:
        data_c: np.array of type complex64 of shape [N]
        
        '''
        
        
        data_sb = integers_to_symbols(data_si=data_si, bits_per_symbol= self.bits_per_symbol)
        if mode == 'explore':
            re, im  = self.sess.run([self.re_sample, self.im_sample], feed_dict={
                    self.input: data_sb              
                })
        if mode == 'exploit':
            re, im  = self.sess.run([self.re_mean, self.im_mean], feed_dict={
                    self.input: data_sb      
                })
        data_c = np.squeeze(re)+1j*np.squeeze(im)
        return data_c
    
    def update(self, preamble_si, actions, labels_si_g, **kwargs):
        """
        Policy update function. Calls self.update_op.

        Inputs:
            inputs: np.array of type integer shape [N] \
                     where each row represents integer representation of symbol
            rewards:np.array of type float shape [N] \
                       The reward corresponding to each action taken (If self.restrict_energy is False,
                        then the term -lambda_p*np.abs(actions) is added to the rewards)
            actions: np.array of type complex64 corresponding to modulated version of each symbol
            stepsize: stepsize for the update operation
                float
        Outputs:
            rewards: the negative of loss given the true and estimated bit streams
        """       
        rewards = -get_bit_l1_loss(labels_si=preamble_si, labels_si_g = labels_si_g,\
                                    bits_per_symbol =self.bits_per_symbol) + 0.5  
        inputs = preamble_si
        if self.restrict_energy is False:
            rewards = rewards - self.lambda_p*np.abs(actions)**2
        if self.restrict_energy is True:
            # rewards_orig = rewards.copy()
            pow_rewards = 1.0 - np.abs(actions)  
            rewards[rewards>=0] = rewards[rewards>=0] + self.lambda_p*pow_rewards[rewards>=0] #Get rewardsantage for power only if got bits right
        data_sb = integers_to_symbols(inputs, self.bits_per_symbol)
        data_c = actions
        
        _, _,re_std, im_std, loss = self.sess.run([self.update_op_mu, self.update_op_sigma, self.re_std, self.im_std, self.surr], feed_dict={
                self.input: data_sb,
                self.actions_re: data_c.real,
                self.actions_im: data_c.imag,
                self.rewards: rewards,
                self.stepsize_mu: self.current_stepsize_mu,
                self.stepsize_sigma: self.current_stepsize_sigma            
        })
        self.times_updated += 1
        
        return -np.average(rewards), re_std, im_std, loss

    def get_std(self):
        return self.sess.run([self.re_std, self.im_std])
       
    
    # def save_model(self, location):
    #     if not os.path.exists(location):
    #         os.makedirs(location)
    #     saver = tf.train.Saver(list(set(self.mu_vars+self.sigma_vars)))
    #     saver.save(self.sess, os.path.join(location,"saved_model"))
    
    # def restore_model(self, location):
    #     saver = tf.train.Saver(list(set(self.mu_vars+self.sigma_vars)))
    #     saver.restore(self.sess, os.path.join(location,"saved_model"))
