import numpy as np
from utils.util_data import get_bit_l1_loss, integers_to_symbols, get_all_unique_symbols, cartesian_2d_to_complex
from utils.visualize import visualize_constellation
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype = torch.float32
np_dtype = np.float32
device = torch.device("cpu")
torch.set_num_threads(1)

class NeuralModel(nn.Module):    
    def __init__(self, 
                 bits_per_symbol,
                 hidden_layers = [40],
                 max_std = 1e1, #
                 min_std = 1e-5, #
                 initial_std=1e-2, #
                 restrict_energy = 1, #0,1,2
                 lambda_prob = 0.1, #
                 activation_fn_hidden = nn.ReLU,
                 activation_fn_output = None,
                 max_amplitude = 0,
                 **kwargs
                 ):
        super(NeuralModel, self).__init__()   


        #MU TRAINING
        assert len(hidden_layers) > 0, "must specify at least one hidden layer"
        layer_dims = [bits_per_symbol]+hidden_layers+[2] # bps --> [hidden layers] --> cartesian 2D
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

        #STD TRAINING 
        self.std_min=torch.tensor(min_std).type(dtype)
        self.std_max=torch.tensor(max_std).type(dtype)
        self.std = nn.Parameter(
            torch.from_numpy(np.array([initial_std, initial_std]).astype(np_dtype)),
            requires_grad=True )

        self.restrict_energy = restrict_energy
        self.lambda_prob = torch.tensor(lambda_prob).type(dtype)
        self.all_unique_symbols = torch.tensor(get_all_unique_symbols(bits_per_symbol=bits_per_symbol)).type(dtype)
        self.bits_per_symbol = bits_per_symbol
        self.max_amplitude = max_amplitude

    def normalize_1(self, means):
        #Get average power
        #WARNING: this can cause memory and speed issues for higher modulation orders like QAM 64000
        avg_power = torch.mean(torch.sum((self.base(self.all_unique_symbols))**2,dim=-1))
        #Get normalization factor based on maximum constraint of 1 on average power
        if self.max_amplitude > 0:
            normalization_factor = torch.sqrt((torch.relu(avg_power-self.max_amplitude)+self.max_amplitude) / self.max_amplitude)
        else:
            normalization_factor = torch.sqrt(torch.relu(avg_power-1.0)+1.0)
        #Divide by normalization factor to get modulated symbols
        means = means/normalization_factor

        return means


    def normalize_2(self, means):
        avg_power = torch.sqrt(torch.mean(torch.sum(means**2,dim=1)))
        normalization = torch.nn.functional.relu(avg_power-1)+1.0
        means = means / normalization
        return means

    def center_means(self, means):
        center = means.mean(dim=0)
        return means - center

    def normalize_center(self, means):
        const_means = self.center_means(self.base(self.all_unique_symbols))
        avg_power = torch.mean(torch.sum(const_means ** 2,dim=-1))
        #Get normalization factor based on maximum constraint of 1 on average power
        if self.max_amplitude > 0:
            normalization_factor = torch.sqrt((torch.relu(avg_power-self.max_amplitude)+self.max_amplitude) / self.max_amplitude)
        else:
            normalization_factor = torch.sqrt(torch.relu(avg_power-1.0)+1.0)
        #Divide by normalization factor to get modulated symbols
        means = self.center_means(means)/normalization_factor
        return means

    def forward(self, input):
        assert len(input.shape) == 2 #input shape should be [N_symbols, bits_per_symbol]
        assert input.shape[1] == self.bits_per_symbol 
        means = self.base(input)
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
        re_logprob = torch.log(self.re_normal.log_prob(actions[:,0]).exp() + self.lambda_prob)
        im_logprob = torch.log(self.im_normal.log_prob(actions[:,1]).exp() + self.lambda_prob)
        return  re_logprob + im_logprob

    def forward_sample(self, input):
        self.forward(input)
        return torch.stack((self.re_normal.sample(), self.im_normal.sample()),1)

    def mu_parameters(self):
        return self.base.parameters()

    def sigma_parameters(self):
        return self.std


class ModulatorNeural():
    def __init__(self,
                 seed=8,
                 bits_per_symbol = 2,
                 lambda_p=.9,
                 stepsize_mu=1e-3,
                 stepsize_sigma=1e-5,
                 restrict_energy=1,
                 max_amplitude=0,
                 lambda_center=0.0,
                 optimizer=torch.optim.Adam,
                **kwargs
                ):
        """
        lambda_p: Scaling factor for power loss term
        restrict_energy: If true normalize outputs(re + 1j*im) to have average energy 1
        """
        torch.manual_seed(seed)


        activations = {
            'lrelu': nn.LeakyReLU,
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }          
        activation_key = kwargs.get('activation_fn_hidden', None)
        if activation_key:
            kwargs['activation_fn_hidden'] = activations[activation_key]


       
        ####################
        # Class Variables
        ####################
        self.mod_class = 'neural'
        self.restrict_energy = restrict_energy
        self.lambda_p = lambda_p
        self.bits_per_symbol = bits_per_symbol
        self.lambda_center = torch.tensor(lambda_center).type(dtype)

        ####################
        # Model
        ####################
        self.model = NeuralModel(
            bits_per_symbol = bits_per_symbol,
            restrict_energy = restrict_energy,max_amplitude= max_amplitude,
            **kwargs)

        #######################
        # For supervised update
        #######################
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        self.epochs = 2**bits_per_symbol

        #######################
        # For unsupervised update
        #######################
        self.surr_loss = lambda adv, logprob: - torch.mean(adv * logprob)
        
        optimizers = {
            'adam': torch.optim.Adam,
            'sgd': torch.optim.SGD,
        }
        if isinstance(optimizer, str):
            optimizer = optimizers[optimizer.lower()]
        if optimizer:
            print("ModulatorNeural initialized with %s optimizer."%optimizer.__name__)
        else:
            print("ModulatorNeural initialized WITHOUT an optimizer")
       
        if optimizer:
            self.optimizer = optimizer([\
                    {'params': self.model.mu_parameters(), 'lr':stepsize_mu},
                    {'params': self.model.sigma_parameters(), 'lr':stepsize_sigma}])
        else:
            self.optimizer = None

    def modulate(self, data_si, mode='explore', detach=True, **kwargs):
        """Modulates data as integers
        Parameters
        ----------
        data_si : np.array, shape [n_samples, ]
            Each sample is an integer representing the symbol
        mode: String that is either 'exploit' or 'explore'. Explore employs sampling, used for training.
              Exploit simply gives the outputs for self.re_mean, self.im_mean for input data_si.
        Returns
        -------

        if detach:
            data_c  : complex np.array, shape [n_samples, ]
        else:
            data_c  : tensor, shape [n_samples, 2] with cartesian 2D form of the complex symbol

            Each sample is modulated into a complex64 point
        """
        if isinstance(data_si, torch.Tensor):
            #input is a tensor of symbols with shape [n_samples, bits_per_symbol]
            assert len(data_si.shape) == 2
            data_s = data_si
        elif isinstance(data_si, np.ndarray):
            #input is an nparray of integers with shape [n_samples, ]
            assert len(data_si.shape) == 1
            data_s = integers_to_symbols(data_si=data_si, bits_per_symbol= self.bits_per_symbol)
            data_s = torch.from_numpy(data_s).type(dtype)
      
        assert data_s.shape[1] == self.bits_per_symbol
        
        if mode == 'exploit': #means for testing/using model
            data_c = self.model.forward(data_s)
        if mode == 'explore': #sampling for training
            data_c = self.model.forward_sample(data_s)
        if detach:
            data_c = data_c.detach().numpy().astype(np.complex64)
            data_c = cartesian_2d_to_complex(data_c)
        return data_c

    def supervised_update(self, data_si, labels_c):
        assert self.optimizer, "ModulatorNeural is not initialized with an optimizer"
        data_s = integers_to_symbols(data_si=data_si, bits_per_symbol= self.bits_per_symbol)
        data_s = torch.from_numpy(data_s).type(dtype)
        #turn complex labels into 2D array
        c_labels = np.stack((labels_c.real.astype(np_dtype), labels_c.imag.astype(np_dtype)), axis=-1)
        #train
        for epoch in range(self.epochs):
            epoch_loss = 0
            for i, d in enumerate(data_s):
                # c_labels[i] is the complex label for the symbol
                c_pred = self.model.forward(d)
                loss = self.loss_fn(c_pred, torch.from_numpy(c_labels[i]))
                epoch_loss = epoch_loss + loss.data[0]
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def update(self, preamble_si, actions, labels_si_g, **kwargs):
        assert self.optimizer, "ModulatorNeural is not initialized with an optimizer"
        """ Policy update function.
        Parameters
        ----------
            labels_si: np.array of type integer shape [n_samples] \
                     where each row represents integer representation of symbol
            actions: np.array of type complex64 corresponding to modulated version of each symbol
            labels_si_g:np.array of type integer shape [n_samples] \
                        The received/estimated possibly erroneous labels which we compare against

            stepsize: float stepsize for the update operation
        """
        preamble_s = integers_to_symbols(data_si=preamble_si, bits_per_symbol= self.bits_per_symbol)
        labels_s_g = integers_to_symbols(data_si=labels_si_g, bits_per_symbol= self.bits_per_symbol)
        preamble_s = torch.from_numpy(preamble_s).type(dtype)


        #rewards

        diff = preamble_si ^ labels_si_g  # xor to find differences in two streams
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
        rewards = torch.from_numpy(rewards).type(dtype)
        logprobs = self.model.forward_log_prob(preamble_s, actions)
        location_loss = torch.norm(torch.mean(self.model.base(self.model.all_unique_symbols))) ** 2
        loss = self.surr_loss(rewards, logprobs) + self.lambda_center * location_loss

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        self.optimizer.step()
        std = self.get_std()
        return -np.average(rewards), std[0], std[1], loss.item()

    def visualize(self, preamble_si, save_plots=False, plots_dir=None, file_name=None, title_prefix=None, title_suffix=None):
        title_string = "Modulator Neural"
        if title_prefix:
            title_string = "%s %s"%(title_prefix, title_string)
        if title_suffix:
            title_string = "%s %s"%(title_string, title_suffix)
        data_m = self.modulate(data_si=preamble_si, mode='explore')
        data_m_centers = self.modulate(data_si=preamble_si, mode='exploit')
        args = {"data":data_m,
                "data_centers":data_m_centers,
                "labels":preamble_si,
                "legend_map":{i:i for i in range(2**self.bits_per_symbol)},
                "title_string":title_string,
                "show":not save_plots}


        visualize_constellation(**args)
        if save_plots:
            if len(file_name.split("."))>2:
                file_name = file_name+".pdf"
            if save_plots:
                plt.savefig("%s/%s"%(plots_dir, file_name))
        plt.close()

    def get_constellation(self):
        data_si = np.arange(2**self.bits_per_symbol)
        data_c = self.modulate(data_si=data_si, mode='exploit')
        return data_c
    
    def get_signal_power(self):
        data_c = self.get_constellation()
        signal_power = np.mean(np.abs(data_c)**2)
        return signal_power
    def get_std(self):
        return self.model.sigma_parameters().data.numpy()

