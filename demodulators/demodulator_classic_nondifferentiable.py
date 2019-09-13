import numpy as np
from utils import util_modulation
from utils.mod_demod_abstract import Demodulator

class DemodulatorClassic(Demodulator):
    def __init__(self, mod_type, block_length = 10000, **kwargs):
        '''
        Initialize parameters used for demodulation 
        Inputs:
        modulation_type: From 'BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64' 
        block_length: block length to break demodulated input into to avoid excessive memory usage
        '''
        self.mod_type = mod_type  
        self.bits_per_symbol, self.symbol_map, self.legend_map = util_modulation.get_modulation_maps(mod_type)
        self.average_energy_per_symbol = np.mean(np.abs(list(self.symbol_map.values()))**2)
        self.M = 2**self.bits_per_symbol
        self.block_length = block_length
        self.demod_class = 'classic'
    
    def demodulate(self, data_c, **kwargs):
        '''
        Inputs:
        data_c: np.array of type complex and shape [n] with modulalated symbols
        Output:
        labels_si_g:  np.array of type integer and shape [n] with estimated labels  
        '''
        means = np.array(list(self.symbol_map.values()))
        #Form matrix of shape [m x 2**self.bits_per_symbol]
        #A: each column contains the complex numbers from data_c (columns identical)
        #B: each row contains the complex numbers corresponding to symbols (rows identical)
        #C = abs(A - B)
        #Labels for each row are determined by which index in row the minimum of that row of C lies
        

        num_blocks = data_c.shape[0]//self.block_length
        if num_blocks == 0:
            num_blocks = 1
        data_c_list = np.array_split(data_c, num_blocks)
        
        labels_si_g = []
        for cur_data_c in data_c_list:                         
            cur_labels_si_g = np.argmin(np.abs(cur_data_c[:, None] - means[None, :]), axis=1)
            labels_si_g.append(cur_labels_si_g)
            
        labels_si_g = np.concatenate(labels_si_g,axis=0)
        return labels_si_g

    def update(self, inputs, actions, data_for_rewards, **kwargs):
        pass

    def demodulate_payload(self, preamble_data_c, preamble_labels_si, payload_data_c):
        '''
        Inputs:
        preamble_data_c: np.array of type complex and shape [n] with modulated preamble
        preamble_labels_si: np.array of type integer and shape [n] with labels for each symbol
        payload_data_c: np.array of type complex and shape [m] with modulated payload
        Output:
        payload_labels_si_g: np.array of type integer and shape [m] with estimated labels for each symbol
                             in payload
        '''
        means = np.array(list(self.symbol_map.values()))
        #Form matrix of shape [m x 2**self.bits_per_symbol]
        #A: each column contains the complex numbers from data_c (columns identical)
        #B: each row contains the complex numbers corresponding to symbols (rows identical)
        #C = abs(A - B)
        #Labels for each row are determined by which index in row the minimum of that row of C lies
        
        num_blocks = payload_data_c.shape[0]//self.block_length
        if num_blocks == 0:
            num_blocks = 1
        payload_data_c_list = np.array_split(payload_data_c, num_blocks)
        
        payload_labels_si_g = []
        for cur_data_c in payload_data_c_list:                         
            cur_labels_si_g = np.argmin(np.abs(cur_data_c[:, None] - means[None, :]), axis=1)
            payload_labels_si_g.append(cur_labels_si_g)
            
        payload_labels_si_g = np.concatenate(payload_labels_si_g,axis=0)
        return payload_labels_si_g

    


######################################################
# Functions for testing
######################################################


def test_single_constellation():
    #Test noisy constellation for single N0
    mod_type_list = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    num_symbols = 1000
    EbN0 = 20.0
    for mod_type in mod_type_list:
        mapper_mod = ModulatorClassic(mod_type=mod_type)
        data_b = get_random_bits(n=num_symbols*mapper_mod.bits_per_symbol)
        data_si = bits_to_integers(data_b=data_b, bits_per_symbol=mapper_mod.bits_per_symbol)
        data_c, labels_si = mapper_mod.modulate(data_si=data_si)
        N0 = mapper_mod.get_N0(EbN0=EbN0)
        noisy_data_c = add_awgn(data_c=data_c, N0=N0)
        visualize_constellation(data=noisy_data_c, labels=labels_si, title_string = 'mod_type: ' + mod_type +
                                    ', EbN0: ' + str(EbN0))

def test_multiple_constellations():
    # #Test noisy constellations for multiple N0s along with demodulator
    num_symbols = 1000
    # mod_type_list = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    mod_type_list = ['QPSK']
    EbN0_array = np.array([0,10,15,20])

    for mod_type in mod_type_list:
        mapper_mod = ModulatorClassic(mod_type=mod_type)    
        mapper_demod = DemodulatorClassic(mod_type=mod_type)
        data_b = get_random_bits(n=num_symbols*mapper_mod.bits_per_symbol)
        data_si = bits_to_integers(data_b=data_b, bits_per_symbol=mapper_mod.bits_per_symbol)
        data_c, labels_si = mapper_mod.modulate(data_si=data_si)

        #Test for multiple N0s at a time
        N0_array = mapper_mod.get_N0(EbN0=EbN0_array)
        noisy_data_c_list = add_awgn(data_c=data_c, N0=N0_array)

        for k,noisy_data_c in enumerate(noisy_data_c_list):
            EbN0 = EbN0_array[k]
            labels_si_g = mapper_demod.demodulate(data_c=noisy_data_c)

            #Visualize correct guesses
            noisy_data_c_correct = noisy_data_c[labels_si==labels_si_g]
            labels_si_correct = labels_si[labels_si==labels_si_g]
            visualize_constellation(data=noisy_data_c_correct, labels=labels_si_correct, title_string = 'Correct: mod_type: ' + mod_type +
                                    ', EbN0: ' + str(EbN0))

            #Visualize incorrect guesses
            noisy_data_c_incorrect = noisy_data_c[labels_si!=labels_si_g]
            labels_si_incorrect = labels_si[labels_si!=labels_si_g]
            visualize_constellation(data=noisy_data_c_incorrect, labels=labels_si_incorrect, title_string = 'Incorrect: mod_type: ' + mod_type +
                                    ', EbN0: ' + str(EbN0))

def test_timing():
    #Demodulator Timing benchmark test
    import time
    num_symbols = int(1e7)
    # mod_type_list = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    mod_type_list = ['QAM64']
    EbN0_array = [0.0,1.0]
    for i,mod_type in enumerate(mod_type_list):
        print(mod_type)
        mapper_mod = ModulatorClassic(mod_type=mod_type)    
        mapper_demod = DemodulatorClassic(mod_type=mod_type)
        data_b = get_random_bits(n=num_symbols*mapper_mod.bits_per_symbol)
        data_si = bits_to_integers(data_b=data_b, bits_per_symbol=mapper_mod.bits_per_symbol)

        start = time.time()
        data_c, labels_si = mapper_mod.modulate(data_si=data_si)
        end = time.time()
        time_taken_1 = end - start
        print("Time taken for modulation: ", time_taken_1)

        #Test for multiple N0s at a time
        N0_array = mapper_mod.get_N0(EbN0=EbN0_array)
        start = time.time()
        noisy_data_c_list = add_awgn(data_c=data_c, N0=N0_array)
        end = time.time()
        time_taken_2 = end - start
        print("Time taken for adding noise : ", time_taken_2)


        for k,noisy_data_c in enumerate(noisy_data_c_list):        
            EbN0 = EbN0_array[k]
            start = time.time()        
            labels_si_g = mapper_demod.demodulate(data_c=noisy_data_c)
            end = time.time()
            time_taken_3 = end - start
            print("Time taken for demodulation: ", time_taken_3)
            start = time .time()
            BER = get_BER(labels_si = labels_si, labels_si_g = labels_si_g, bits_per_symbol=mapper_mod.bits_per_symbol)
            end = time.time()
            time_taken_4 = end - start
            print("Time taken for BER calculation: ", time_taken_4)

def test_BER():
    import matplotlib.pyplot as plt
    #Demodulator BER test
    num_symbols = int(1e7)
    mod_type_list = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    EbN0_array = np.arange(0,20,2)

    BER_array = np.empty([len(mod_type_list), EbN0_array.shape[0]])
    for i,mod_type in enumerate(mod_type_list):
        print("Processing mod_type: " + mod_type)
        mapper_mod = ModulatorClassic(mod_type=mod_type)    
        mapper_demod = DemodulatorClassic(mod_type=mod_type)
        data_b = get_random_bits(n=num_symbols*mapper_mod.bits_per_symbol) 
        data_si = bits_to_integers(data_b=data_b, bits_per_symbol=mapper_mod.bits_per_symbol)
        data_c, labels_si = mapper_mod.modulate(data_si=data_si)
        #Test for multiple N0s at a time
        N0_array = mapper_mod.get_N0(EbN0=EbN0_array)   
        noisy_data_c_list = add_awgn(data_c=data_c, N0=N0_array)   
        for k,noisy_data_c in enumerate(noisy_data_c_list):        
            EbN0 = EbN0_array[k]       
            labels_si_g = mapper_demod.demodulate(data_c=noisy_data_c)
            BER = get_BER(labels_si = labels_si, labels_si_g = labels_si_g, bits_per_symbol=mapper_mod.bits_per_symbol)
            BER_array[i,k] = BER
#             print("Mod type: ", mod_type, "EbN0:", EbN0, "BER: ", BER)
    
    for i in range(BER_array.shape[0]):
        plt.plot(EbN0_array,BER_array[i,:], 'o-', label = mod_type_list[i])
    
    plt.yscale('log')
    plt.ylabel('BER')
    plt.xlabel('EbN0(dB)')
    plt.title('BER vs EbN0')
    plt.legend(loc = 'upper right')
    plt.show()
    
    
def main():
    from util_data import get_random_bits, add_awgn, get_BER, bits_to_integers
    from visualize import visualize_constellation, gen_demod_grid
    from modulator_classic import ModulatorClassic
    '''Contains function calls for various tests'''

#     test_single_constellation()  
#     test_multiple_constellations()
#     test_timing()
    # test_BER()
    
if __name__ == "__main__":
    main()

