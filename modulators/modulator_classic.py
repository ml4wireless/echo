import numpy as np
import torch
# from utils import util_modulation
import os
import pickle
from utils import util_modulation
from utils.mod_demod_abstract import Modulator
from utils.util_data import cartesian_2d_to_complex

dtype = torch.float32
np_dtype = np.float32

def to_numpy(x):
    #Convert pytorch tensor to numpy array
    if x is not None:
        return x.data.numpy().astype(np_dtype)

def to_tensor(x):
    #Convert numpy array to pytorch tensor
    if x is not None:
        return torch.from_numpy(x).type(dtype)



class ModulatorClassic():
    def __init__(self, bits_per_symbol, max_amplitude = 0, **kwargs):
        '''
        Initialize variables used for modulation
        bits_per_symbol: decides what scheme to use (2 for QPSK, 3 for 8PSK, etc.)
        REMOVED rotate angle: the angle in rads that the constellation is rotated by clockwise
        '''
        self.bits_per_symbol = bits_per_symbol
        self.M = 2**self.bits_per_symbol
        self.symbol_map = to_tensor(util_modulation.get_symbol_map(bits_per_symbol=bits_per_symbol)).float()  
#         self.symbol_map = rotate_clockwise(self.symbol_map, rotate_angle)
        self.mod_class = 'classic' #Classic modulator
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
        
    def modulate(self, data_si, detach = True, **kwargs):
        if isinstance(data_si, np.ndarray):
            data_si = to_tensor(data_si)
        data_si = data_si.long()
        data_c = self.symbol_map[data_si[:],:]
        if detach:
            data_c = to_numpy(data_c).astype(np.complex64)
            data_c = cartesian_2d_to_complex(data_c)
        return data_c
    
    def update(self, **kwargs):
        pass

    def get_std(self):
        return 0
    
    def get_signal_power(self):
        data_c = self.get_constellation()
        signal_power = np.mean(np.abs(data_c)**2)
        return signal_power

    def get_constellation(self):
        data_si = np.arange(2**self.bits_per_symbol)
        data_c = self.modulate(data_si=data_si, mode='exploit')
        return data_c

    def visualize(self, preamble_si):
        data_m = self.modulate(data_si=preamble_si, mode='explore')
        data_m_centers = self.modulate(data_si=preamble_si, mode='exploit')
        args = {"data":data_m,
                "data_centers":data_m_centers,
                "labels":preamble_si,
                "legend_map":{i:i for i in range(2**self.bits_per_symbol)},
                "title_string":'Modulator Classic',
                "show":True}
        visualize_constellation(**args)
