import numpy as np
from utils.mod_demod_abstract import Demodulator
from utils.util_data import torch_tensor_to_numpy as to_numpy
from utils.util_data import get_bit_l1_loss, cartesian_2d_to_complex
from utils.visualize import visualize_decision_boundary
import torch
# import matplotlib.pyplot as plt

class DemodulatorCluster(Demodulator):
    '''
    Implements cluster based demod where each class is represented solely by its centre and demdodulation occurs 
    based on nearest centre.    
    '''
    def __init__(self, bits_per_symbol, decay_rate, explore_prob, block_size = 1000, **kwargs):
        '''
        Inputs:
        decay_rate: Rate at which to decay rewards from previous training sets
        bits_per_symbol: Number of constellation points is 2**bits_per_symbol
        explore_prob: Probability to decide demodulation uniform randomly from all choices     
        block_size: Break up into blocks before demodulation to not use up too much memory
        '''
        
        self.bits_per_symbol = bits_per_symbol
        self.num_classes = 2**bits_per_symbol
        self.decay_rate = decay_rate
        self.explore_prob = explore_prob
        self.demod_class = 'cluster'
        
        self.cluster_means = np.random.uniform(low=-5, high=5,size=self.num_classes) + 1j*np.random.uniform(low=-5, high=5, size = self.num_classes)    
        self.num_updates = np.zeros(self.num_classes, dtype = 'float')#Stores weighted number of updates to mean so far
        self.block_size = block_size
        
        
    def demodulate(self, data_c, mode = 'explore', **kwargs):
        '''
        Inputs:
        data_c: np.array of type complex and shape [N] corresponding to  modulated symbols
        mode: Make random choices for exploration if in explore mode
        Ouput:
        labels_si_g: np.array of type integer and shape [N] corresponding to integere representation of demodulated symbols 
        '''
        # print(data_c)
        #TODO break this into blocks
        if isinstance(data_c, torch.Tensor):
            data_c = to_numpy(data_c)
        elif isinstance(data_c, np.ndarray):
            pass
        else:
            print("Warning demodulator_cluster")
        #Find distance to cluster_means
        dist = np.abs(data_c[:,None] - self.cluster_means[None,:])
        labels_si_g = np.argmin(dist, axis=1)        
        if mode == 'explore':
            indices = np.where(np.random.uniform(size=data_c.shape[0]) < self.explore_prob)           
            labels_si_g[indices] = np.random.randint(low=0,high=self.num_classes, size = indices[0].shape[0])
            
        elif mode == 'exploit':
            pass
        else:
            raise ValueError('Unknown mode ' + str(mode))
            
        return labels_si_g
        
        
    def update(self, preamble_si, labels_si_g, data_c, **kwargs):
        inputs = data_c
        # actions = self.demodulate(data_for_rewards, mode='exploit')
        rewards = 1.0 / (get_bit_l1_loss(labels_si=preamble_si, labels_si_g = labels_si_g,\
                                        bits_per_symbol =self.bits_per_symbol) + 1e-2)
        if np.min(rewards) < 0:
            raise ValueError('Received minimum reward of ' + str(np.min(rewards)) + ' while rewards must be non-negative!')
    
        for action_type in np.unique(labels_si_g):
            indices = np.where(labels_si_g==action_type)
            cur_inputs = inputs[indices]
            # cur_actions = actions[indices]
            cur_rewards = rewards[indices]
            reward_sum = np.sum(cur_rewards)
            num_updates_decay = self.decay_rate*self.num_updates[action_type]
            self.cluster_means[action_type] *= num_updates_decay 
            self.cluster_means[action_type] += np.sum(cur_inputs*cur_rewards)
            self.cluster_means[action_type] /= num_updates_decay + reward_sum
            self.num_updates[action_type] = num_updates_decay + reward_sum        



    def get_demod_grid(self,grid_2d):
        grid_2d = np.reshape(to_numpy(grid_2d), (-1, 2))
        grid_2d = cartesian_2d_to_complex(grid_2d)
        labels_si_g= self.demodulate(data_c=grid_2d, mode = 'exploit')
        return labels_si_g


    # def visualize(self, save_plots=False, plots_dir=None, file_name=None, title_prefix=None, title_suffix=None):
    #     title_string = "Demodulator Cluster"
    #     if title_prefix:
    #         title_string = "%s %s"%(title_prefix, title_string)
    #     if title_suffix:
    #         title_string = "%s %s"%(title_string, title_suffix)
    #     args = {"points_per_dim":100,
    #             "legend_map":{i:i for i in range(2**self.bits_per_symbol)},
    #             "title_string":title_string,
    #             "show":not save_plots}
    #     visualize_decision_boundary(self, **args)()
    #     if len(file_name.split("."))>2:
    #         file_name = file_name+".pdf"
    #     if save_plots:
    #         plt.savefig("%s/%s"%(plots_dir, file_name))
    #     plt.close()

        
