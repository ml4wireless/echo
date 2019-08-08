import numpy as np
import torch
from utils import util_modulation

class Classic():
    def __init__(self,*, 
                bits_per_symbol:int, 
                max_amplitude:float = 0.0,
                block_length:int = 10000, 
                **kwargs):
        '''
        Initialize parameters used for demodulation 
        Inputs:
        bits_per_symbol: decides what scheme to use (2 for QPSK, 3 for 8PSK, etc.)
        block_length: block length to break demodulated input into to avoid excessive memory usage
        '''
        self.name = 'classic'
        self.model_type = 'demodulator'
        self.bits_per_symbol = bits_per_symbol
        self.block_length = block_length
        self.symbol_map = torch.from_numpy(util_modulation.get_symbol_map(bits_per_symbol=bits_per_symbol)).float()  
        self.max_amplitude = max_amplitude
        self.normalize_symbol_map()
        
          
    
    def normalize_symbol_map(self):
        #I and Q separate
        if self.max_amplitude > 0.0:
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
    

    def forward(self, symbols:torch.Tensor, **kwargs):
        '''
        Inputs:
        symbols: torch.tensor of type float and shape [n,2] with modulalated symbols
        soft: Bool either True or False
        Output:        
        if mode = "prob"
            labels_si_g: torch.tensor of type float and shape [n,2] with probabilities for each demodulated symbol
        else:
            labels_si_g: torch.tensor of type int and shape [n,1] with integer representation of demodulated symbols
        '''
            
        num_blocks = np.ceil(symbols.shape[0]/self.block_length).astype(np.int32)
        if num_blocks == 0:
            num_blocks = 1
            
        blocks_list = []
        eff_block_length = symbols.shape[0]//num_blocks
        for i in range(num_blocks):
            blocks_list.append(symbols[i*eff_block_length:(i+1)*eff_block_length, :])  

        logits = []
        for block in blocks_list:     
            dist = torch.sum((block[:,None] - self.symbol_map[None,:])**2, dim = -1)
            block_logits = -dist #We give logits here since pytorch loss function applies the softmax
            logits.append(block_logits)
       
      
        logits = torch.cat(logits, dim=0)
        return logits

    __call__ = forward

    def update(self, signal, true_symbols, **kwargs):
        pass
    