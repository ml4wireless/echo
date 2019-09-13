#Implements neural network based demodulator class
import numpy as np
import os
import tensorflow as tf
from utils.mod_demod_abstract import Demodulator
from utils.util_tf import fancy_slice_2d, normc_initializer, sample_discrete_logits
from utils.util_data import complex_to_cartesian_2d, get_complex_l2_loss, get_cluster_loss

class DemodulatorNeural(Demodulator):
    '''Implements neural net receiver'''
    def __init__(self,
                 seed=7,
                 hidden_layers = [16],
                 bits_per_symbol = 2,
                 activation_fn_hidden = tf.nn.relu,
                 kernel_initializer_hidden = normc_initializer(1.0),
                 bias_initializer_hidden =  tf.glorot_uniform_initializer(),
                 activation_fn_output = None,
                 kernel_initializer_output = normc_initializer(1.0),
                 bias_initializer_output =  tf.glorot_uniform_initializer(),
                 optimizer = tf.train.AdamOptimizer,
                 initial_eps = 1e-1,
                 max_eps = 2e-1,
                 min_eps = 1e-4,
                 lambda_prob = 1e-1,
                 loss_type='l2',
                 explore_prob=0.5,
                 strong_supervision_prob=0.,
                 stepsize_mu=1e-2,
                 stepsize_eps=1e-5,
                 stepsize_cross_entropy=1e-3,
                 cross_entropy_weight=1.0,
                 **kwargs
                ):
        '''
        Inputs:
        seed: Tensor flow graph level seed (default = 7)
        hidden_layers: A list of length [num_layers] with entries corresponding to 
                       number of hidden units in each layer (Default = [16])
        bits_per_symbol: Determines number of units in output layer as 2**bits_per_symbol
        activation_fn_hidden: Activation function to be used for hidden layers (default = tf.nn.relu) strings also accepted
        kernel_initializer_hidden:  Kernel intitializer for hidden layers (default = normc_initializer(1.0)) 
        bias_initializer_hidden: Bias initialize for hidden layers (default = tf.glorot_uniform_initializer())
        activation_fn_output: Activation function to be used for output layer (default = None)
        kernel_initializer_output: Kernel intitializer for output layer (default = normc_initializer(1.0))
        bias_initializer_output: Bias initializer for output layer (default = tf.glorot_uniform_initializer())
        optimizer: Optimizer to be used while training (default = tf.train.AdamOptimizer),
        initial_eps: Initial probability for exploring each class
        min_eps: Minimum probability for exploring each class
        max_eps: Maximum probability for exploring each class
        lambda_prob: Regularizer for log probability 
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

        self.loss_type = loss_type
        self.explore_prob=explore_prob
        self.strong_supervision_prob=strong_supervision_prob
        self.current_stepsize_mu=stepsize_mu
        self.current_stepsize_eps=stepsize_eps
        self.current_stepsize_cross_entropy=stepsize_cross_entropy
        self.current_cross_entropy_weight=cross_entropy_weight

        #Define graph
        self.graph = tf.Graph()
        self.demod_class = "neural"
        with self.graph.as_default():
            tf.set_random_seed(seed) #Set seed
            
            self.bits_per_symbol = bits_per_symbol
            self.num_classes = 2**self.bits_per_symbol
            self.eps = tf.Variable(initial_eps)
            self.min_eps = min_eps
            self.max_eps = max_eps
            ####################################################
            # Inputs to actions
            ####################################################
            #Placeholders 
            self.input = tf.placeholder(shape=[None,2], dtype=tf.float32) #2D cartesian input

            #Define hidden layers
            net = self.input
            global_layer_num = 0
            for cur_layer_num in range(len(hidden_layers)):
                cur_layer_name = 'mu/layer' + str(global_layer_num)  
                global_layer_num += 1
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
            self.logits = tf.layers.dense(
                          inputs = net,
                          units = 2**bits_per_symbol,
                          activation = activation_fn_output,
                          kernel_initializer = kernel_initializer_output,
                          bias_initializer = bias_initializer_output,
                          name = cur_layer_name
                          ) 
            self.softmax_logits = tf.nn.softmax(self.logits)          
            
            #Probabilities and actions
            self.eff_eps = tf.maximum(self.eps, self.min_eps)
            self.eff_eps = tf.minimum(self.eff_eps, self.max_eps)
            self.probs = self.eff_eps/float(self.num_classes) + (1 - self.eff_eps)*self.softmax_logits #Log probablities calculated from tensors
            
            #Actions sampled according to probability distribution self.probs
            self.sampled_actions = sample_discrete_logits(tf.log(self.probs))

            #Optimal actions sampling the index with maximum probability in self.probs            
            self.optimal_actions = tf.argmax(self.logits, axis = 1) 
            
            
            
            mu_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'mu/')
            eps_vars = [self.eps]
            cross_entropy_vars = list(mu_vars)  # Python 2.7 compatible copy
            self.mu_vars = mu_vars
            self.eps_vars = eps_vars
            self.cross_entropy_vars = cross_entropy_vars
        
            #########################################################
            # Loss calculation            
            #########################################################            
            
            #Placeholders for loss calculation
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32) #The actions chosen by NN. 
                                                 #Here an array of integers betwee (0, 2**num_bits)            
            self.adv = tf.placeholder(tf.float32, [None]) # advantages for gradient computation
                
            self.stepsize_mu = tf.placeholder(shape=[], dtype=tf.float32)
            self.stepsize_eps = tf.placeholder(shape=[], dtype=tf.float32)
            self.stepsize_cross_entropy = tf.placeholder(shape=[], dtype=tf.float32)

            #Calculate log probablities of selected actions
            num_samples = tf.shape(self.input)[0]
            self.selected_probs = fancy_slice_2d(X=self.probs, inds0=tf.range(num_samples),
                                                    inds1= self.actions) 
            self.selected_logprobs = tf.log(lambda_prob + self.selected_probs)

            #Define loss
            self.loss_vec = -self.adv*self.selected_logprobs
            self.loss  = tf.reduce_mean(self.loss_vec)
            self.cross_entropy_signal = tf.placeholder(shape=[None], dtype=tf.int32)
            self.cross_entropy_weight = tf.placeholder(shape=[], dtype=tf.float32)
            self.cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.cross_entropy_signal, logits=self.logits))
            
            self.update_op_mu = optimizer(self.stepsize_mu).minimize(self.loss, var_list=mu_vars)
            self.update_op_eps = optimizer(self.stepsize_eps).minimize(self.loss, var_list=eps_vars)
            self.cross_entropy_op = optimizer(self.stepsize_cross_entropy).minimize(self.cross_entropy_weight*self.cross_entropy_loss, var_list=cross_entropy_vars)

            #Define session
            self.sess = tf.Session()
            
            #Initialize all variables
            self.sess.run(tf.global_variables_initializer())
           
            
    def demodulate(self, data_c, mode='explore', **kwargs):
        '''
        Demodulated complex numbers into integer representation of symbols
        If mode is explore demodulate based on exploration policy
        If mode is exploit demodulate and return symbols that are most likely based on current NN
        Inputs:
        data_c: np.array of type complex of shape [N.] containing modulated signal
        mode: String must be either 'explore' or 'exploit'    
        Output:
        labels_si_g: np.array of type integer and shape [N] containing integer representation of symbols
        '''
        
        data_d = complex_to_cartesian_2d(data_c=data_c)
        if mode == 'explore':           
            if np.random.uniform()<self.explore_prob: 
                labels_si_g = self.sess.run(self.sampled_actions, feed_dict = {self.input:data_d})                     
            else:
                labels_si_g = self.sess.run(self.optimal_actions, feed_dict = {self.input:data_d})     
        elif mode == 'exploit':
            labels_si_g = self.sess.run(self.optimal_actions, feed_dict = {self.input:data_d})
        elif mode == 'proba':
            labels_si_g = self.sess.run(self.softmax_logits, feed_dict = {self.input:data_d})
        else:
            raise ValueError('Mode: ' + str(mode) + ' must be either "explore" or "exploit"')
     
        return labels_si_g
    
    def update(self, inputs, actions, data_for_rewards, **kwargs):

        if 'mode' not in kwargs:
            raise ValueError("Mode not found")

        elif kwargs['mode'].lower() == 'echo_echo' and self.cross_entropy_weight == 0:
            if self.loss_type == 'cluster':
                rewards = -get_cluster_loss(data_c=inputs, data_c_g=data_for_rewards, k=2**self.bits_per_symbol) 
            elif self.loss_type == 'l2':
                rewards = -get_complex_l2_loss(data_c=inputs,data_c_g=data_for_rewards)
            else:
                raise ValueError('Unknown demod loss type ' + str(self.loss_type))     
            return self.policy_update(inputs, actions, rewards)

        elif kwargs['mode'].lower() == 'echo':
            return self.cross_entropy_update(data_p=inputs, data_m=data_for_rewards)


    
    def policy_update(self, data_c, actions, adv, **kwargs):
        """
        Policy update function. Calls self.update_op.

        Inputs:
            data_c: np.array of type complex of shape [N] corresponding to modulated signal
            actions: np.array of type integer and shape [N] containing symbols corresponding to actions taken
            stepsize_mu: stepsize for the update operation for nn
            stepsize_eps: stepsize for update of exploration probability
            adv: np.array of type float and shape [N] containing advantages/rewards corresponding to each action
        Outputs:
            avg_loss: scalar: average loss given the true and estimated symbol streams

        """
        data_d = complex_to_cartesian_2d(data_c=data_c)
        _,_,eps,loss =  self.sess.run( [self.update_op_mu,self.update_op_eps,self.eps,self.loss], \
                feed_dict={
                self.input: data_d,
                self.actions: actions,               
                self.adv: adv,
                self.stepsize_mu: self.current_stepsize_mu,
                self.stepsize_eps: self.current_stepsize_eps
                })
        avg_reward = np.average(adv)  
        return -avg_reward, eps, loss
    
    def cross_entropy_update(self, data_p, data_m, **kwargs):
        """
        Second loss function to provide stronger signal for learning.
            ** (previously named short_circuit_update) **
        
        Inputs:
            data_p: np.array of type int of shape [N] with symbols as integers
            data_m: np.array of type complex of shape [N] corresponding to modulated data_p
            stepsize: float to define step size for gradient updates
        Outputs:
        """
        data_d = complex_to_cartesian_2d(data_c=data_m)
        self.sess.run(self.cross_entropy_op, feed_dict={self.input: data_d,
                                                        self.cross_entropy_signal: data_p,
                                                        self.stepsize_cross_entropy: self.current_stepsize_cross_entropy,
                                                        self.cross_entropy_weight: self.current_stepsize_cross_entropy})
    
    # def save_model(self, location):
    #     if not os.path.exists(location):
    #         os.makedirs(location)
    #     saver = tf.train.Saver(list(set(self.mu_vars + self.eps_vars + self.cross_entropy_vars)))
    #     saver.save(self.sess, os.path.join(location,"saved_model"))
    
    # def restore_model(self, location):
    #     saver = tf.train.Saver(list(set(self.mu_vars + self.eps_vars + self.cross_entropy_vars)))
    #     saver.restore(self.sess, os.path.join(location,"saved_model"))


    def get_std(self):
        return self.sess.run(self.eff_eps)
            
