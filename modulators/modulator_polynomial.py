import numpy as np
from utils.util_data import get_bit_l1_loss, integers_to_symbols, cartesian_2d_to_complex, get_all_unique_symbols
import torch
from torch import nn
from torch.autograd import Variable
from itertools import chain, combinations
from utils.util_data import torch_tensor_to_numpy as to_numpy, numpy_to_torch_tensor as to_tensor

dtype = torch.float32
np_dtype = np.float32
device = torch.device("cpu")
torch.set_num_threads(1)

def build_exp(bps):
    if bps < 1:
        return []
    elif bps == 1:
        return [[0], [1]]
    else:
        ret = []
        for l in build_exp(bps-1):
            ret += [l + [0]]
            ret += [l + [1]]
        return ret


class PolynomialModel(nn.Module):
    def __init__(self, bits_per_symbol, initial_std, std_min, std_max, restrict_energy):
        super(PolynomialModel, self).__init__()
        self.bits_per_symbol = bits_per_symbol
        self.linear = nn.Linear(2**bits_per_symbol, 2, bias=False)
        def _init_weights(m):
            if type(m) == nn.Linear:
                y = 1.0/np.sqrt(m.in_features)
                m.weight.data.uniform_(-y, y)
        self.linear.apply(_init_weights) 
        self.std_min=torch.tensor(std_min).type(dtype)
        self.std_max=torch.tensor(std_max).type(dtype)
        self.std = nn.Parameter(
            torch.from_numpy(np.array([initial_std, initial_std]).astype(np_dtype)),
            requires_grad=True
        )
        #self.exp is used to builds polynomial features of the symbols i.e. [b1, b2] ---> [b1, b2, b1b2, 1]. 
        #We ignore b1**2 and b2**2 and higher order polynomial features because bits are 0 or 1
        self.exp = torch.from_numpy(np.array(build_exp(bits_per_symbol))).float() 
        self.all_unique_symbols = torch.tensor(get_all_unique_symbols(bits_per_symbol=bits_per_symbol)).type(dtype)
        self.restrict_energy = restrict_energy

    def polynomial(self, input):
        return torch.stack([torch.prod(torch.pow(input.float(), exp.float()), 1) for exp in self.exp]).t()

    def QAM16(self, input):
        coeff = torch.tensor([[-1, 0, 0, 0, 2/3, 0,0,0,2,0,0,0,-4/3,0,0,0],[-1, 2/3, 2, -4/3, 0,0,0,0,0,0,0,0,0,0,0,0]]).float().t()
        # print(self.polynomial(input).shape, coeff.shape)
        means = torch.mm(self.polynomial(input), coeff)
        return means

    def normalize_1(self, means):
        #Get average power        
        #WARNING: this can cause memory and speed issues for higher modulation orders like QAM 64000
        poly_unique_symbols = self.polynomial(self.all_unique_symbols)
        avg_power = torch.mean(torch.sum((self.linear(poly_unique_symbols))**2,dim=-1)) 
        #Get normalization factor based on maximum constraint of 1 on average power
        normalization_factor = torch.sqrt(torch.relu(avg_power-1.0)+1.0)       
        #Divide by normalization factor to get modulated symbols
        return means/normalization_factor 

    def normalize_2(self, means):
        avg_power = torch.sqrt(torch.mean(torch.sum(means**2,dim=1)))
        normalization = torch.nn.functional.relu(avg_power-1)+1.0 
        means = means / normalization
        return means

    def center_means(self, means):
        center = means.mean(dim=0)
        return means - center

    def normalize_center(self, means):
        poly_unique_symbols = self.polynomial(self.all_unique_symbols)
        avg_power = torch.mean(torch.sum(self.center_means(self.linear(poly_unique_symbols))**2,dim=-1))
        normalization = torch.nn.functional.relu(avg_power-1)+1.0
        means = self.center_means(means) / normalization
        return means
        
    def forward(self, input):
        means = self.linear(self.polynomial(input))
        # means = self.QAM16(input) #ANSWER FOR QAM16
        ###################
        # Normalize outputs
        ################### 
        if (self.restrict_energy == 1):
            means = self.normalize_1(means)
        elif (self.restrict_energy == 2):
            means = self.normalize_2(means)
        elif (self.restrict_energy == 3):
            means = self.normalize_center(means)
        self.std_bounded = nn.functional.relu(self.std-self.std_min)+self.std_min
        self.std_bounded = self.std_max-nn.functional.relu(-(self.std_bounded-self.std_max))
        self.re_normal = torch.distributions.normal.Normal(means[:,0], self.std_bounded[0])
        self.im_normal = torch.distributions.normal.Normal(means[:,1], self.std_bounded[1])
        return means
    
    def forward_log_prob(self, input, actions): 
        self.forward(input)
        return self.re_normal.log_prob(actions[:,0]) + self.im_normal.log_prob(actions[:,1])
    
    def forward_sample(self, input):
        self.forward(input)
        return torch.stack((self.re_normal.sample(), self.im_normal.sample()),1)

    def l1_loss(self):
        return torch.norm(self.linear.weight, p=1) + torch.sum(torch.abs(self.std))

    def mu_parameters(self):
        return self.linear.parameters()

    def sigma_parameters(self):
        return self.std


class ModulatorPolynomial():
    def __init__(self,
                 seed=8,
                 bits_per_symbol = 2,
                 restrict_energy = 3,
                 lambda_p=.9,
                 initial_std=0.0,
                 min_std=1e-1,
                 max_std=100,
                 stepsize_mu=0.0,
                 stepsize_sigma=0.0,
                 lambda_l1 = 0.0,
                 optimizer=torch.optim.Adam,
                **kwargs
                ):
        """
        lambda_p: Scaling factor for power loss term
        restrict_energy: If true normalize outputs(re + 1j*im) to have average energy 1 
        """
        torch.manual_seed(seed)
        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }
        if isinstance(optimizer, str):
            optimizer = optimizers[optimizer.lower()]
        if optimizer:
            print("ModulatorPolynomial initialized with %s optimizer."%optimizer.__name__)
        else:
            print("ModulatorPolynomial initialized WITHOUT an optimizer")
        ####################
        # Class Variables
        ####################
        self.mod_class = 'polynomial'
        self.restrict_energy = restrict_energy
        self.lambda_p = lambda_p
        self.bits_per_symbol = bits_per_symbol
        self.lambda_l1 = torch.tensor(lambda_l1).float()
        
        ####################
        # Model
        ####################
        self.model = PolynomialModel(bits_per_symbol, initial_std, min_std,max_std, self.restrict_energy)
        
        #######################
        # For supervised update
        #######################
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.epochs = 2**bits_per_symbol
        
        #######################
        # For unsupervised update
        #######################
        self.surr_loss = lambda adv, logprob: - torch.mean(adv * logprob) 
        # self.optimizer = torch.optim.Adam(self.model.parameters())
        if optimizer:
            self.optimizer = optimizer([\
                    {'params': self.model.mu_parameters(), 'lr':stepsize_mu},
                    {'params': self.model.sigma_parameters(), 'lr':stepsize_sigma}])
        else:
            self.optimizer = None
        
    def make_features(self, X):
        """Builds polynomial features of the symbols i.e. [b1, b2] = [b1, b2, b1b2, 1]. 
        We ignore b1**2 and b2**2 and higher order polynomial features because bits are 0 or 1
        Parameters
        ----------
        X : array-like, shape [n_samples, bits_per_symbol]
            The data to transform, row by row.
        Returns
        -------
        XP : np.ndarray shape [n_samples, NP]
            The matrix of features, where NP (= 2**bits_per_symbol) is the number of polynomial
            features generated from the combination of inputs.
        """
        n_samples, n_features = X.shape
        # allocate output data
        XP = np.empty((n_samples, 2**self.bits_per_symbol), dtype=np_dtype)
        combs = chain.from_iterable(combinations(range(n_features), i) for i in range(self.bits_per_symbol+1))
        for i, c in enumerate(combs):
            XP[:, i] = X[:, c].prod(1)
        return XP
    

    def modulate(self, data_si, mode='explore', detach=True, **kwargs):
        """Modulates data as integers using polynomial features
        Parameters
        ----------
        data_si : np.array, shape [n_samples, ]
            Each sample is an integer representing the symbol
        mode: String that is either 'exploit' or 'explore'. Explore employs sampling, used for training. 
              Exploit simply gives the outputs for self.re_mean, self.im_mean for input data_si.
        Returns
        -------
        data_c  : complex np.array, shape [n_samples, ]
            Each sample is modulated into a complex64 point
        """

        if isinstance(data_si, torch.Tensor):
            #input is a tensor of symbols
            # print(data_si)
            # raise TypeError("ModulatorPolynomial.modulate does not take data_si torch.Tensor input.")
            data_si = np.uint32(to_numpy(data_si))
            data_s = torch.from_numpy(integers_to_symbols(data_si=data_si, bits_per_symbol= self.bits_per_symbol))
        elif isinstance(data_si, np.ndarray):
            #input is an nparray of integers
            data_s = to_tensor(integers_to_symbols(data_si=data_si, bits_per_symbol= self.bits_per_symbol))
       
        if mode == 'exploit': #means for testing/using model
            data_c = self.model.forward(data_s)
        if mode == 'explore': #sampling for training
            data_c = self.model.forward_sample(data_s)
        if mode == 'QAM16':
            data_c = self.model.QAM16(data_s)
        
        if detach:
            data_c = data_c.detach().numpy().astype(np.complex64)
            data_c = cartesian_2d_to_complex(data_c)
        return data_c

    def supervised_update(self, data_si, labels_c):
        assert self.optimizer, "ModulatorPolynomial is not initialized with an optimizer"
        data_s = to_tensor(integers_to_symbols(data_si=data_si, bits_per_symbol= self.bits_per_symbol))
        #turn complex labels into 2D array
        c_labels = np.stack((labels_c.real.astype(np_dtype), labels_c.imag.astype(np_dtype)), axis=-1)
        #train                       
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i, d in enumerate(data_s):
                # here p is the polynomial featurized symbol.
                # c_labels[i] is the complex label for the symbol
                c_pred = self.model.forward(d)
                loss = self.loss_fn(c_pred, torch.from_numpy(c_labels[i]))
                epoch_loss = epoch_loss + loss.data[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def update(self, preamble_si, actions, labels_si_g, **kwargs):
        assert self.optimizer, "ModulatorPolynomial is not initialized with an optimizer"
        """ Policy update function.
        Parameters
        ----------
            labels_si: np.array of type integer shape [n_samples] \
                     where each row represents integer representation of symbol
            labels_si_g:np.array of type integer shape [n_samples] \
                        The received/estimated possibly erroneous labels which we compare against
            data_c: np.array of type complex64 corresponding to modulated version of each symbol
            stepsize: float stepsize for the update operation
        """
        preamble_s = to_tensor(integers_to_symbols(data_si=preamble_si, bits_per_symbol= self.bits_per_symbol))
        #labels_s_g = integers_to_symbols(data_si=labels_si_g, bits_per_symbol= self.bits_per_symbol)

        #rewards
        rewards = -get_bit_l1_loss(labels_si=preamble_si, labels_si_g=labels_si_g, bits_per_symbol=self.bits_per_symbol) + 0.5
        if self.restrict_energy is 0: # penalize for high power
            power = np.abs(actions)**2
            rewards = rewards - self.lambda_p*power
        # if self.restrict_energy is True: # give good actions extra reward for not using the full power 
        #     unused_power = 1.0 - np.abs(actions)  
        #     rewards[rewards>=0] = rewards[rewards>=0] + self.lambda_p*unused_power[rewards>=0]


        #turn complex actions into 2D array for real component and imaginary component
        actions = torch.from_numpy(np.stack((actions.real.astype(np_dtype), actions.imag.astype(np_dtype)), axis=-1))
        
        #compute loss
        rewards = torch.from_numpy(rewards)
        logprobs = self.model.forward_log_prob(preamble_s, actions)
        loss = self.surr_loss(rewards.double(), logprobs.double())
        loss += self.lambda_l1.double() * self.model.l1_loss().double() #add l1 regularization
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        
        self.optimizer.step()
        std = self.get_std()
        return -np.average(rewards), std[0], std[1], loss.item()

    
    def get_std(self):
        return self.model.sigma_parameters().data

    def get_constellation(self):
        data_si = np.arange(2**self.bits_per_symbol)
        data_c = self.modulate(data_si=data_si, mode='exploit')
        return data_c
       