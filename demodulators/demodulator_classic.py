import numpy as np
import torch
from utils import util_modulation
from utils.mod_demod_abstract import Demodulator
from utils.visualize import visualize_decision_boundary
from utils.util_data import complex_to_cartesian_2d

dtype = torch.float32

def to_tensor(x):
    #Convert numpy array to pytorch tensor
    if x is not None:
        return torch.from_numpy(x).type(dtype)

class DemodulatorClassic(Demodulator):
    def __init__(self, bits_per_symbol, max_amplitude = 0,block_length = 10000, rotate_angle = 0, **kwargs):
        '''
        Initialize parameters used for demodulation 
        Inputs:
        bits_per_symbol: decides what scheme to use (2 for QPSK, 3 for 8PSK, etc.)
        block_length: block length to break demodulated input into to avoid excessive memory usage
        rotate angle: the angle in rads that the constellation is rotated by clockwise
        '''
        self.bits_per_symbol = bits_per_symbol
        self.M = 2**self.bits_per_symbol
        self.block_length = block_length
        self.symbol_map = to_tensor(util_modulation.get_symbol_map(bits_per_symbol=bits_per_symbol)).float()  
        self.demod_class = 'classic'   
        self.max_amplitude = max_amplitude
        self.normalize_symbol_map()
        
          
    
    def normalize_symbol_map(self):
        #I and Q separate
        if self.max_amplitude > 0:
            avg_power = torch.mean(torch.sum(self.symbol_map**2,dim=-1))
            normalization_factor = torch.sqrt((torch.relu(avg_power-self.max_amplitude)+self.max_amplitude) / self.max_amplitude)
            self.symbol_map = self.symbol_map/normalization_factor
            
        #Based on I^2 + Q^2
#         if self.max_amplitude > 0:
#             max_abs = torch.max(torch.norm(self.symbol_map,dim=1))
# #             print(self.symbol_map)
# #             print("max_abs", max_abs)
#             if max_abs > self.max_amplitude:
#                 self.symbol_map = (self.symbol_map/max_abs)*self.max_amplitude
#         print(self.symbol_map)
    def demodulate(self, data_c, mode = "exploit", detach = True):
        '''
        Inputs:
        data_c: torch.tensor of type float and shape [n,2] with modulalated symbols
        soft: Bool either True or False
        Output:        
        if mode = "prob"
            labels_si_g: torch.tensor of type float and shape [n,2] with probabilities for each demodulated symbol
        else:
            labels_si_g: torch.tensor of type int and shape [n,1] with integer representation of demodulated symbols
        '''
        if isinstance(data_c, np.ndarray):
            data_c = to_tensor(complex_to_cartesian_2d(data_c=data_c))
            
        num_blocks = np.ceil(data_c.shape[0]/self.block_length).astype(np.int32)
        if num_blocks == 0:
            num_blocks = 1
            
        data_c_list = []
        eff_block_length = data_c.shape[0]//num_blocks
        for i in range(num_blocks):
            data_c_list.append(data_c[i*eff_block_length:(i+1)*eff_block_length, :])  

        labels_si_g = []
        if mode == 'logit':
            for cur_data_c in data_c_list:     
                dist = torch.sum((cur_data_c[:,None] - self.symbol_map[None,:])**2, dim = -1)
                cur_labels_si_g = -dist #We give logits here since pytorch loss function applies the softmax
                labels_si_g.append(cur_labels_si_g)
        else:
            for cur_data_c in data_c_list:     
                dist = torch.sum((cur_data_c[:,None] - self.symbol_map[None,:])**2, dim = -1)
                cur_labels_si_g = torch.argmin(dist, dim=-1)
                labels_si_g.append(cur_labels_si_g)
      
        labels_si_g = torch.cat(labels_si_g, dim=0)
        if detach:
            labels_si_g=labels_si_g.detach().numpy().astype(int)
        return labels_si_g
    
    def update(self, **kwargs):
        pass
    
    def cross_entropy_update(self, **kwargs):
        pass
    
    def visualize(self):
        visualize_decision_boundary(self,  points_per_dim=100, title_string="Demodulator Classic")()


    def get_demod_grid(self, grid_2d):
        return None
