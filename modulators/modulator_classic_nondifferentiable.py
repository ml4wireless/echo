###################################################################################################
# Defines class for  single sample per symbol modulator based on traditional modulation schemes

# Dependencies
# util_modulation.py
# util_data.py
###################################################################################################

import numpy as np
# from utils import util_modulation
import os
import pickle
from utils import util_modulation
from utils.mod_demod_abstract import Modulator


#Classic 1 sample per symbol unrealistic ideal radio
class ModulatorClassic(Modulator):
    def __init__(self, mod_type, **kwargs):
        '''
        Initialize parameters used for modulation 
        Inputs:
        modulation_type: From 'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64'       
        '''
        self.mod_type = mod_type  
        self.bits_per_symbol, self.symbol_map, self.legend_map = util_modulation.get_modulation_maps(mod_type)
        self.average_energy_per_symbol = np.mean(np.abs(list(self.symbol_map.values()))**2)
        self.M = 2**self.bits_per_symbol
        self.mod_class = 'classic' #Classic modulator
        self.times_updated = 0
        
    def modulate(self, data_si, **kwargs):
        '''
        Modulates given bit stream into stream of complex numbers
        Inputs:
        data: np.array of type integer. Should contain symbols if is_bitstream = False, and should be a bitstream if otherwise.
        is_bitstream: boolean indicating data type
        mode: Should be none for classic modulator. Used by neural network modulator to distinguish training from testing. 
        
        Output:
        data_c: np.array of type complex containing modulated symbols as to I + jQ
        ''' 
        #Get symbol array from symbol map
        symbol_array = np.array(list(self.symbol_map.values()))
        #Fancy index into this array     
        data_c = symbol_array[data_si]
        return data_c

    def update(self, preamble_si, actions, labels_si_g, **kwargs):
        pass


    # def save_model(self, location):
    #     if not os.path.exists(location):
    #         os.makedirs(location)

    #     pickling_on = open(os.path.join(location, "save_model.pickle"),"wb")
    #     pickle.dump(self, pickling_on)
    #     pickling_on.close()

    # def restore_model(location):
    #     if not os.path.exists(location):
    #         os.makedirs(location)

    #     pickling_off = open(os.path.join(location, "save_model.pickle"),"rb")
    #     return pickle.load(pickling_off)
       
    def get_N0(self,EbN0):
        '''
        Returns N0 to acheive target EbN0 based on average energy of constellation
        '''
        EbN0 = np.array(EbN0)
        EbN0_lin = 10**(0.1*EbN0)
        N0 = self.average_energy_per_symbol/(EbN0_lin*self.bits_per_symbol)
        return N0
    def get_constellation(self):
        return None
    
    def get_std(self):
        return None
  
    
#Test function for modulator class
def main():    
    import visualize
    import util_data    
    num_symbols = int(1e7)
    mod_types = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    for mod_type in mod_types:
        print("Mod type: " + mod_type)        
        mapper = ModulatorClassic(mod_type=mod_type)
        data_b = util_data.get_random_bits(n=num_symbols*mapper.bits_per_symbol)
        data_si = util_data.bits_to_integers(data_b=data_b, bits_per_symbol=mapper.bits_per_symbol)
        data_c, labels_si = mapper.modulate(data_si=data_si)
        visualize.visualize_constellation(data=data_c, labels=labels_si, title_string=mod_type)
        
if __name__ == "__main__":
    main()