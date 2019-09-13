import numpy as np
# import argparse
import torch
import math
# from sklearn.preprocessing import PolynomialFeatures
# import torch.autograd
# import torch.nn.functional as F
from torch import nn
from utils.util_data import complex_to_cartesian_2d, get_complex_l2_loss, get_cluster_loss, cartesian_2d_to_complex
from utils.util_data import torch_tensor_to_numpy as to_numpy, numpy_to_torch_tensor as to_tensor
from utils.util_torch import StraightThroughArgMaxLayer, SoftKArgMaxLayer

dtype = torch.float32
np_dtype = np.float32
device = torch.device("cpu")
torch.set_num_threads(1)

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


class PolynomialModel(nn.Module):
    def __init__(self, num_symbols, bits_per_symbol, degree_polynomial): #initial_logstd, std_min, restrict_energy):
        super(PolynomialModel, self).__init__()
        
        self.num_polynomial_terms =int( nCr(degree_polynomial + 2, degree_polynomial) )
        #self.exp is the exponents used to builds polynomial features of the 2-D cartesian inputs 
        self.exp = to_tensor(np.array([[j, i-j] for i in range(degree_polynomial+1) for j in range(i+1)]))
        assert len(self.exp) == self.num_polynomial_terms, "for each term (i.e. x^1*y^2) there should be a pair of exponents (i.e. [1,2]) in self.exp"
        self.base = nn.Linear(self.num_polynomial_terms, num_symbols, bias=False) #polynomial input
        def _init_weights(m):
            if type(m) == nn.Linear:
                y = 1.0/np.sqrt(m.in_features)
                m.weight.data.uniform_(-y, y)
        self.base.apply(_init_weights) 
        self.softmax = nn.Softmax(1)

    def polynomial(self, input):
        return torch.stack([torch.prod(torch.pow(input.float(), exp.float()), 1) for exp in self.exp]).t()
   
    def forward_logits(self, input):
        assert len(input.shape) == 2, "input should be 2D cartesian points with shape [n_samples, 2]"
        #2D cartesian input --> polynomial --> logit output (# classes = 2**bits_per_symbol)
        logits = self.base(self.polynomial(input))
        return logits

    #Return probalities of classes 
    def forward(self, input):
        return self.softmax(self.forward_logits(input))
    
    #Return optimal actions of distribution (argmax of probabilities)
    def forward_optimal_action(self, input):
        probs = self.forward(input)
        _, optimal_actions = torch.max(probs, 1) 
        return optimal_actions
    
    #Return actions sampled from probabilities
    def forward_sampled_action(self, input):
        #Samples from probabilities
        probs = self.forward(input)
        m = torch.distributions.Categorical(probs)
        samples = m.sample()
        return samples
    
    #Return logprobability of an action
    def forward_log_probs(self, input, actions):
        probs = self.forward(input)
        selected_log_probs = torch.log(probs[torch.arange(probs.size()[0]),actions])
        return selected_log_probs

    def l1_loss(self):
        return torch.norm(self.base.weight, p=1)

class DemodulatorPolynomial():
    def __init__(self,
                 seed=1,
                 bits_per_symbol = 2,
                 degree_polynomial = 3,
                 loss_type='l2',
                 stepsize_cross_entropy = 1e-3,
                 stepsize_policy_grad = 1e-2,
                 cross_entropy_weight=1.0, #
                 lambda_l1=0.0,
                 optimizer=torch.optim.Adam,
                 **kwargs
                ):
        torch.manual_seed(seed)
        
        self.loss_type = loss_type
        self.bits_per_symbol = bits_per_symbol   
        self.degree_polynomial = degree_polynomial
        # self.poly_featurizer = PolynomialFeatures(degree_polynomial)
        
        self.model = PolynomialModel(2**bits_per_symbol, bits_per_symbol, degree_polynomial)
        self.demod_class = 'polynomial'
        self.stepsize_cross_entropy=stepsize_cross_entropy
        self.lambda_l1 = torch.tensor(lambda_l1).float()
        
        
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
            print("DemodulatorPolynomial initialized with %s optimizer."%optimizer.__name__)
        else:
            print("DemodulatorPolynomial initialized WITHOUT an optimizer")

        if optimizer:
            self.policy_gradient_optimizer =  optimizer(self.model.base.parameters(), lr=stepsize_policy_grad)
            self.cross_entropy_weight = torch.tensor(cross_entropy_weight).type(dtype)
            self.cross_entropy_optimizer = optimizer(self.model.base.parameters(), lr=stepsize_cross_entropy)
        else:
            self.policy_gradient_optimizer = None
            self.cross_entropy_optimizer = None
        
    def demodulate(self, data_c, mode='explore', detach=True, **kwargs):
        """Demodulated complex numbers into integer representation of symbols using polynomial features
        Parameters
        ----------
        data_c : complex np.array, shape [n_samples, ]
            Each sample is a complex number of the modulated signal
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
        elif isinstance(data_c, np.ndarray): #input is a np array of complex inputs
            assert len(data_c.shape)==1, "input must be np.array of complex inputs of shape [n_samples, ]"
            data_d = torch.from_numpy(complex_to_cartesian_2d(data_c=data_c))
        else:
            raise ValueError("DemodulatorPolynomial.demodulate data type of 'data_c' is unrecognized.")
        
        if mode == 'explore': 
            # integer/long array, shape [n_samples, ]
            assert detach, "must detach in 'explore' mode"      
            labels_si_g = self.model.forward_sampled_action(data_d)
        elif mode == 'exploit':
            # integer/long array, shape [n_samples, ] 
            assert detach, "must detach in 'exploit' mode"
            labels_si_g = self.model.forward_optimal_action(data_d)
        elif mode == 'logit':
            # float array, shape [n_samples, num_unique_symbols (=2**bits_per_symbol)] 
            labels_si_g = self.model.forward_logits(data_d)
        elif mode == 'prob':
            # float array, shape [n_samples, num_unique_symbols (=2**bits_per_symbol)]
            labels_si_g = self.model.forward(data_d)
        elif mode == 'prob_bits':
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
            labels_si_g = labels_si_g.detach().numpy().astype(int)

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
        assert self.cross_entropy_optimizer, "DemodulatorPolynomial is not initialized with an optimizer"
        #Cartesian Input
        data_d = torch.from_numpy(complex_to_cartesian_2d(data_c=data_c))
        #train                       
        output = self.model.forward_logits(data_d)
        target = torch.from_numpy(labels_si)
        criterion = nn.CrossEntropyLoss()
        loss = self.cross_entropy_weight * torch.mean(criterion(output, target))
        loss += self.lambda_l1 * self.model.l1_loss() #add l1 regularization
        
        #Backprop
        self.cross_entropy_optimizer.zero_grad()
        loss.backward()
        self.cross_entropy_optimizer.step()
        #######################
        # For supervised update
        #######################
        # loss_fn = torch.nn.NLLLoss()
        # data_d = complex_to_cartesian_2d(data_c=data_c)
        # #train                       
        # pred_si = self.model.forward(data_d) #softmax vector
        # loss = loss_fn(pred_si, torch.from_numpy(labels_si))
        # l1_regularization,l2_regularization = (0,0) 
        # for param in self.model.parameters():
        #     # l1_regularization += param.norm(1)
        #     l2_regularization += param.norm(2)
        # # l1_regularization = torch.tensor(self.lambda_l1) * l1_regularization
        # l2_regularization = torch.tensor(self.lambda_l2) * l2_regularization
        # loss = loss + l1_regularization + l2_regularization
        # loss.backward()
        # self.optimizer.step()
    
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
        assert self.policy_gradient_optimizer, "DemodulatorPolynomial is not initialized with an optimizer"
        data_d = complex_to_cartesian_2d(data_c=data_c)
        actions = torch.from_numpy(actions) #guesses of the symbols
        selected_logprobs = self.model.forward_log_probs(data_d, actions) #what were the logprob of the guesses made
        
        reward = torch.from_numpy(reward)
        loss = -torch.mean(selected_logprobs * reward)
        loss += self.lambda_l1 * self.model.l1_loss() #add l1 regularization
        
        #backprop
        self.policy_gradient_optimizer.zero_grad()
        loss.backward()
        self.policy_gradient_optimizer.step()
        
        return -np.average(reward), loss.item()

    def get_demod_grid(self,grid_2d):
        grid_2d = np.reshape(to_numpy(grid_2d), (-1, 2))
        grid_2d = cartesian_2d_to_complex(grid_2d)
        labels_si_g= self.demodulate(data_c=grid_2d, mode = 'exploit')
        return labels_si_g
    

        
