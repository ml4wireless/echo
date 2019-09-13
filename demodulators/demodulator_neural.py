import numpy as np
import torch
import math
from torch import nn
from torch.distributions import Categorical #, OneHotCategorical
from utils.util_data import complex_to_cartesian_2d, get_complex_l2_loss, get_cluster_loss, integers_to_symbols
from utils.util_torch import StraightThroughArgMaxLayer, SoftKArgMaxLayer
from utils.visualize import visualize_decision_boundary

import matplotlib.pyplot as plt

dtype = torch.float32
np_dtype = np.float32
device = torch.device("cpu")
torch.set_num_threads(1)

#<--- notargmaxed <***
#<--- notargmaxed <***
#<--- argmax      <---
#<--- notargmaxed <***
 
# * simulated gradient ; <-- backprop gradient that gets updated)

                                                  

class NeuralModel(nn.Module):
    def __init__(self, 
                 num_classes, 
                 hidden_layers = [16],
                 activation_fn_hidden = nn.ReLU,
                 activation_fn_output = None,
                 initial_eps = 1e-1,
                 max_eps = 2e-1,
                 min_eps = 1e-4,
                 **kwargs):
        super(NeuralModel, self).__init__()

        assert len(hidden_layers) > 0, "must specify at least one hidden layer"
        #BASE MODEL'S LAYER DIMENSIONS: cartesian 2D,...hidden layer dims..., num_classes
        layer_dims = [2]+hidden_layers+[num_classes] 
        layers = [] 
        for i in range(len(layer_dims)-1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2: #hidden layers
                layers.append(activation_fn_hidden())
            elif activation_fn_output:  #output layer
                layers.append(activation_fn_output())
        self.base = nn.Sequential(*layers) 
        def _init_weights(m):
            if type(m) == nn.Linear:
                y = 1.0/np.sqrt(m.in_features)
                m.weight.data.uniform_(-y, y)
                m.bias.data.fill_(0.01)
        self.base.apply(_init_weights) 

        #EPSILON -- for increased exploration
        self.explore = initial_eps > 0
        self.eps = nn.Parameter(
            torch.tensor(initial_eps).type(dtype),
            requires_grad=True
        )
        self.min_eps = torch.tensor(min_eps).type(dtype)
        self.max_eps = torch.tensor(max_eps).type(dtype)
        self.num_classes = torch.tensor(num_classes).type(dtype)
        
        self.softmax = nn.Softmax(1)
   
    def logits(self, input):
        assert len(input.shape) == 2, "input should be 2D cartesian points with shape [n_samples, 2]"
        return self.base(input.float())

    def forward(self, input):
        if self.explore:
            eps = torch.min(torch.max(self.eps, self.min_eps), self.max_eps)
            return eps/self.num_classes + (1 - eps) * self.softmax(self.logits(input))
        else:
            return self.softmax(self.logits(input))

    # #Return optimal actions of distribution (max of logits)
    # def forward_optimal_actions(self, input):
    #     _, optimal_actions = torch.max(self.logits(input), 1) 
    #     return optimal_actions
    
    # #Return actions sampled from distribution
    # def forward_sample_actions(self, input):
    #     #Samples from probabilities
    #     probs = self.forward(input)
    #     m = torch.distributions.Categorical(probs)
    #     samples = m.sample()
    #     return samples

    # #Calculate log probablities of selected actions
    # def forward_log_probs(self, input, actions):
    #     probs = self.forward(input)
    #     selected = probs[torch.arange(logprobs.size()[0]),actions]
    #     selected_log_probs = nn.Log(self.lambda_prob.expand(selected.size()) + selected)
    #     return selected_log_probs

class DemodulatorNeural():
    def __init__(self,
                 seed=1,
                 bits_per_symbol = 2,
                 loss_type='l2', #
                 explore_prob=0.5, #
                 stepsize_cross_entropy = 1e-3, #
                 stepsize_mu=1e-2, #
                 stepsize_eps=1e-5, #
                 optimizer = torch.optim.Adam, #
                 cross_entropy_weight=1.0, #
                 **kwargs
                ):
        torch.manual_seed(seed)
        print('NEURAL DEMOD ', kwargs['activation_fn_hidden'])
        activations = {
            'lrelu': nn.LeakyReLU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }          
        activation_key = kwargs.get('activation_fn_hidden', None)
        if activation_key:
            kwargs['activation_fn_hidden'] = activations[activation_key]
        
        self.demod_class = 'neural'
        self.loss_type = loss_type
        self.explore_prob = explore_prob
        self.bits_per_symbol = bits_per_symbol
        self.model = NeuralModel(
                 num_classes=2**bits_per_symbol, 
                 **kwargs
        )
        #######################
        # Optimizer
        #######################
        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }
        if isinstance(optimizer, str):
            optimizer = optimizers[optimizer.lower()]
        if optimizer:
            print("DemodulatorNeural initialized with %s optimizer."%optimizer.__name__)
        else:
            print("DemodulatorNeural initialized WITHOUT an optimizer")

        if optimizer:
            self.policy_gradient_optimizer = optimizer(
                [ {'params': self.model.base.parameters(), 'lr': stepsize_mu},
                  {'params': [self.model.eps], 'lr': stepsize_eps}
                ])
            self.cross_entropy_weight = torch.tensor(cross_entropy_weight).type(dtype)
            self.cross_entropy_optimizer = optimizer(self.model.base.parameters(), lr=stepsize_cross_entropy)
        else:
            self.policy_gradient_optimizer = None
            self.cross_entropy_optimizer = None

    def demodulate(self, data_c, mode='explore', detach=True):
        """Demodulate complex/cartesian numbers into integer representation of symbols 
        Parameters
        ----------
        data_c : complex np.array, shape [n_samples, ] OR float torch.Tensor, shape [n_samples, 2]
            Each sample is a complex number (or cartesian pair) of the modulated signal
        mode    : String must be either 'explore' or 'exploit'. If mode is explore demodulate based on exploration policy. 
               If mode is exploit demodulate and return symbols that are most likely based on the current model
        Returns
        -------
        data_si  : integer np.array, shape [n_samples, ]
            Integer representation of symbols
        """
        if isinstance(data_c, torch.Tensor): #input is a tensor of cartesian inputs
            assert len(data_c.shape)==2, "input must be tensor of cartesian inputs of shape [n_samples, 2]"
            data_d = data_c
        elif isinstance(data_c, np.ndarray):
            assert len(data_c.shape)==1, "input must be np.array of complex inputs of shape [n_samples, ]"
            data_d = torch.from_numpy(complex_to_cartesian_2d(data_c=data_c))
        else:
            raise ValueError("DemodulatorNeural.demodulate data type of 'data_c' is unrecognized.")
           
        random_flip = np.random.uniform() < self.explore_prob
        if mode == 'explore' and random_flip:  
            # integer/long array, shape [n_samples, ] 
            assert detach, "must detach in 'explore' mode"
            probs = self.model.forward(data_d)
            m = Categorical(probs)
            actions = m.sample()       
            labels_si_g = actions
        elif mode == 'exploit' or (mode == 'explore' and not random_flip):
            # integer/long array, shape [n_samples, ] 
            assert detach, "must detach in 'exploit'/'explore' mode"
            probs = self.model.forward(data_d)
            _, actions = torch.max(probs, 1) 
            labels_si_g = actions
        elif mode == 'logit':
            # float array, shape [n_samples, num_unique_symbols (=2**bits_per_symbol)]
            labels_si_g = self.model.logits(data_d)
        elif mode == 'prob':
            # float array, shape [n_samples, num_unique_symbols (=2**bits_per_symbol)]
            #softmax logits directly to not include exploration probability
            labels_si_g = nn.functional.softmax(self.model.logits(data_d), dim=1)
        elif mode == 'prob_bits':
            # TO DO: transfer this code into an experiment main.py
            # float array, shape [n_samples, bits_per_symbol]
            #partial bits weighted by probabilites outputted for each symbol
            probs = self.model.forward(data_d)
            actions = probs
            integers = np.array([i for i in range(2**self.bits_per_symbol)])
            symbols = integers_to_symbols(integers, bits_per_symbol=self.bits_per_symbol)
            symbols = symbols.T
            columns = []
            for bit_row in symbols:
                bit_row = torch.from_numpy(bit_row.reshape(len(bit_row), 1)).type(dtype)
                columns += [torch.mm(actions, bit_row).squeeze(1)]
            labels_si_g = torch.stack(columns, 1)
        elif mode == 'differentiable':
            # TO DO: transfer this code into an experiement main.py 
            # float array, shape [n_samples, bits_per_symbol]
            probs = self.model.forward(data_d)
            argmax = StraightThroughArgMaxLayer() #returns one-hot encoding
            actions = argmax(probs)
            integers = np.array([i for i in range(2**self.bits_per_symbol)])
            symbols = integers_to_symbols(integers, bits_per_symbol=self.bits_per_symbol)
            symbols = symbols.T
            columns = []
            for bit_row in symbols:
                bit_row = torch.from_numpy(bit_row.reshape(len(bit_row), 1)).type(dtype)
                columns += [torch.mm(actions, bit_row).squeeze(1)]
            labels_si_g = torch.stack(columns, 1)    
        else:
            print("Mode = {}".format(mode))
            raise ValueError('Mode: is unrecognized')
       
        if detach:
                labels_si_g=labels_si_g.detach().numpy().astype(int)
        return labels_si_g

    def update(self, inputs, actions, data_for_rewards, **kwargs):
        if 'mode' not in kwargs:
            raise ValueError("Mode not found")
        elif kwargs['mode'].lower() == 'echo_echo' and self.stepsize_cross_entropy == 0:
            if self.loss_type == 'cluster':
                rewards = -get_cluster_loss(data_c=inputs, data_c_g=data_for_rewards, k=2**self.bits_per_symbol) 
            elif self.loss_type == 'l2':
                rewards = -get_complex_l2_loss(data_c=inputs,data_c_g=data_for_rewards)
            else:
                raise ValueError('Unknown demod loss type ' + str(self.loss_type))     
            return self.policy_update(inputs, actions, rewards)
        elif kwargs['mode'].lower() == 'echo':
            return self.cross_entropy_update(data_c=data_for_rewards, labels_si=inputs)

                
    def cross_entropy_update(self, data_c, labels_si, **kwargs):
        assert self.cross_entropy_optimizer, "DemodulatorNeural is not initialized with an optimizer"
        #Cartesian Input
        data_d = torch.from_numpy(complex_to_cartesian_2d(data_c=data_c))
        #train                       
        output = self.model.logits(data_d)
        target = torch.from_numpy(labels_si)
        criterion = nn.CrossEntropyLoss()
        loss = self.cross_entropy_weight * torch.mean(criterion(output, target))
        #Backprop
        self.cross_entropy_optimizer.zero_grad()
        loss.backward()
        self.cross_entropy_optimizer.step()
    
    def policy_update(self, data_c, actions, reward):
        """ Policy update function.
        Parameters
        ----------
            data_c: complex np.array, shape [n_samples,] \
                     Modulated signal
            actions: integer np.array, shape [n_samples,] \
                        Guesses of symbols
            labels_si: integer np.array, shape [n_samples,]
                        True symbols
            stepsize: float stepsize for the update operation
        Returns
        ----------
            avg_loss: scalar
                    average loss given the true and estimated symbol streams
        """
        assert self.policy_gradient_optimizer, "DemodulatorNeural is not initialized with an optimizer"
        data_d = torch.from_numpy(complex_to_cartesian_2d(data_c=data_c))
        actions = torch.from_numpy(actions) #guesses of the symbols
        
        probs = self.model.forward(data_d)
        m = Categorical(probs)

        reward = torch.from_numpy(reward).type(dtype)
        loss = -torch.mean(m.log_prob(actions) * reward)
        
        #Backprop
        self.policy_gradient_optimizer.zero_grad()
        loss.backward()
        self.policy_gradient_optimizer.step()
        return -np.average(reward), self.model.eps.item(), loss.item()

    def visualize(self, save_plots=False, plots_dir=None, file_name=None, title_prefix=None, title_suffix=None):
        title_string = "Demodulator Neural"
        if title_prefix:
            title_string = "%s %s"%(title_prefix, title_string)
        if title_suffix:
            title_string = "%s %s"%(title_string, title_suffix)
        args = {"points_per_dim":100,
                "legend_map":{i:i for i in range(2**self.bits_per_symbol)},
                "title_string":title_string,
                "show":not save_plots}
        visualize_decision_boundary(self, **args)()

        if save_plots:
            if len(file_name.split("."))>2:
                file_name = file_name+".pdf"
            if save_plots:
                plt.savefig("%s/%s"%(plots_dir, file_name))
        plt.close()

    def get_demod_grid(self,grid_2d):
        labels_si_g= self.demodulate(data_c=grid_2d, mode = 'exploit')
        return labels_si_g
