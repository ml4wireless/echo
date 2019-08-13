import torch

from utils import util_modulation
from utils.util_data import symbols_to_integers


class Classic():
    def __init__(self,*, 
        bits_per_symbol, 
        max_amplitude = 0.0, 
        **kwargs):
        '''
        Initialize variables used for modulation
        bits_per_symbol: decides what scheme to use (2 for QPSK, 3 for 8PSK, etc.)
        '''
        self.name = 'classic'
        self.model_type = 'modulator'
        self.bits_per_symbol = bits_per_symbol
        self.M = 2**self.bits_per_symbol
        self.symbol_map = torch.from_numpy(util_modulation.get_symbol_map(bits_per_symbol=bits_per_symbol)).float()  
        self.mod_class = 'classic' #Classic modulator
        self.max_amplitude = max_amplitude
        
        self.normalize_symbol_map()
        
    
    def normalize_symbol_map(self):
        #I and Q separate
        if self.max_amplitude > 0:
            avg_power = torch.mean(torch.sum(self.symbol_map**2,dim=-1))
            normalization_factor = torch.sqrt((torch.relu(avg_power-self.max_amplitude)+self.max_amplitude) / self.max_amplitude)
            self.symbol_map = self.symbol_map/normalization_factor
            
    
    #input bit symbols torch.Tensor 
    #output cartesian points torch.Tensor
    def forward(self, symbols:torch.Tensor) -> torch.Tensor:
        data_si = symbols_to_integers(symbols)
        data_si = data_si.long()
        cartesian_points = self.symbol_map[data_si[:],:]
        return cartesian_points
    
    __call__ = forward
    
    def update(self, **kwargs):
        pass