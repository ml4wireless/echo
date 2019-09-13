import numpy as np
import os
import pickle
import torch
from utils.visualize import visualize_constellation
from utils.util_data import get_bit_l1_loss
from utils.mod_demod_abstract import Modulator
# import matplotlib.pyplot as plt
class SingleEntryMod():
    '''
    This class maintains single entry of a lookup table and supports the following functions:
    update_training_set()
    get_mean()   
    '''
    def __init__(self, decay_rate, k):
        '''
        Inputs:
        decay_rate: Rate at which to decay rewards from previous training sets
        k: Maintain top k rewards along with corresponding actions       
        '''
        self.training_set = None #Numpy array with first column containing actions and second 
                                            #column containing rewards
        self.decay_rate = decay_rate
        self.k = k
        self.default_mean_action = np.random.uniform(low=-5,high=5) + 1j*np.random.uniform(low=-5,high=5)
        
    def update_training_set(self, actions, rewards):
        '''
        Inputs:
        actions: np.array of type complex and shape [N] corresponding to actions taken
        rewards: np.array of type float and shape [N] with rewards for the actions. Assume rewards are non-negative
        '''
        if np.min(rewards) < 0:
            raise ValueError('Received minimum reward of ' + str(np.min(rewards)) + ' while rewards must be non-negative!')
        
        
        new_training_set = np.hstack([actions[:,None], rewards[:,None]])
        if self.training_set is not None:
            self.training_set[:,1] *= self.decay_rate
            combined_training_set = np.vstack([self.training_set, new_training_set])
        else:
            combined_training_set = new_training_set
               
        sorted_indices = np.argsort(-np.abs(combined_training_set[:,1]))
        combined_training_set_sorted =combined_training_set[sorted_indices,:]
        self.training_set = combined_training_set_sorted[:self.k,:]
    
    def get_mean_action(self):
        '''
        Returns mean of actions in training set weighted by rewards
        '''
        if self.training_set is not None:
            mean_action = np.sum(self.training_set[:,0]*self.training_set[:,1])/np.sum(self.training_set[:,1])
        else:
            mean_action = self.default_mean_action
        return mean_action
        
    

class ModulatorCluster(Modulator):
    '''
    Implements class for cluster mean based modulator. For each input symbol the output is determined by a class object
    of type SingleEntry    
    '''
    
    def __init__(self, bits_per_symbol, decay_rate, std_decay_rate, k, initial_std, lambda_p = 0, restrict_energy=True, **kwargs):
        '''
        Inputs:
        decay_rate: Rate at which to decay rewards from previous training sets
        k: Maintain top k rewards along with corresponding actions  
        bits_per_symbol: Number of constellation points is 2**bits_per_symbol
        std: Standard deviation while exploring during modulation
        restrict_energy: If true average power of constellation set to 1
        lambda_p: Penalty for using power
        '''
        self.mod_class = 'cluster'
        self.num_constellation_points = 2**bits_per_symbol
        self.bits_per_symbol = bits_per_symbol
        self.decay_rate = decay_rate
        self.std_decay_rate = std_decay_rate
        self.k  = k
        self.initial_std = initial_std
        self.std = initial_std
        self.restrict_energy = restrict_energy
        self.single_entries = []  
        self.lambda_p = lambda_p
        for i in range(self.num_constellation_points):
            self.single_entries.append(SingleEntryMod(decay_rate=decay_rate, k=k))         
    
    def modulate(self, data_si, mode='explore', std=None, **kwargs):
        # print(self.std)
        if std is None:
            scale = self.std
        else:
            scale = std
        means = self.get_mean_actions()  
        
        if mode == 'explore':
            data_c = np.random.normal(loc=means[data_si].real, scale = scale) + \
                          1j*np.random.normal(loc=means[data_si].imag, scale = scale)
        elif mode == 'exploit':
            data_c = means[data_si]
        else:
            raise ValueError('Unknown mode ' + str(mode))
        return data_c

    def update(self, preamble_si, actions, labels_si_g, **kwargs):
        '''
        Inputs:
        inputs: np.array of type integer and shape [N] containing integer representation of symbols 
        actions: np.array of type complex and shape [N] containing complex modulated symbols
        rewards: np.array of type float containing reward for the actions (Rewards must be non-negative)
        '''

        rewards = 1.0 / (get_bit_l1_loss(labels_si=preamble_si, labels_si_g = labels_si_g,\
                                    bits_per_symbol =self.bits_per_symbol) + 1e-2)
        # print(np.average(rewards))
        # print(self.std)
        if np.min(rewards) < 0:
            raise ValueError('Received minimum reward of ' + str(np.min(rewards)) + ' while rewards must be non-negative!')
        
        # if self.restrict_energy is False:
        #     rewards = rewards + 1./(reward_lambda + self.lambda_p*np.abs(actions))

        for input_type in np.unique(preamble_si):
            indices = np.where(preamble_si==input_type)
            # cur_inputs = inputs[indices]
            cur_actions = actions[indices]
            cur_rewards = rewards[indices]
            self.single_entries[input_type].update_training_set(actions=cur_actions, rewards=cur_rewards)

        std = (self.std, self.std)
        self.std *= self.std_decay_rate


        # mimicking neural mod update return format 'return -np.average(rewards), std[0], std[1], loss.item()'
        return -np.average(rewards), std[0], std[1], 0

                
    # def visualize(self, preamble_si, save_plots=False, plots_dir=None, file_name=None, title_prefix=None, title_suffix=None):
    #     title_string = "Modulator Cluster"
    #     if title_prefix:
    #         title_string = "%s %s"%(title_prefix, title_string)
    #     if title_suffix:
    #         title_string = "%s %s"%(title_string, title_suffix)
    #     data_m = self.modulate(data_si=preamble_si, mode='explore')
    #     data_m_centers = self.modulate(data_si=preamble_si, mode='exploit')
    #     args = {"data":data_m,
    #             "data_centers":data_m_centers,
    #             "labels":preamble_si,
    #             "legend_map":{i:i for i in range(2**self.bits_per_symbol)},
    #             "title_string":title_string,
    #             "show":not save_plots}
    #     visualize_constellation(**args)
    #     if len(file_name.split("."))>2:
    #         file_name = file_name+".pdf"
    #     if save_plots:
    #         plt.savefig("%s/%s"%(plots_dir, file_name))
    #     plt.close()

    def get_constellation(self):
        data_si = np.arange(2**self.bits_per_symbol)
        data_c = self.modulate(data_si=data_si, mode='exploit')
        return data_c


    def get_std(self):
        return torch.tensor((self.std, self.std))

    def get_mean_actions(self):
            means = np.array([s.get_mean_action() for s in self.single_entries])        
            if self.restrict_energy is True:
                avg_means = np.mean(np.abs(means)**2)
                if avg_means >= 1:
                    means /= np.sqrt(avg_means)                
            return means
        