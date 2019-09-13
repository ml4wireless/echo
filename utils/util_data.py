# coding: utf-8

# In[1]:


# Contains functions for data generation and manipulations

import numpy as np
import torch
import sys
from utils.kmeans import Kmeans
import itertools



##############################################################################
# Adding Noise
##############################################################################
def get_awgn(N0, n):
    '''
    N0: Noise power
    n: The shape of the tensor returned is [n,2] 
    Each entry is i.i.d Gaussian with mean 0 and standard deviation np.sqrt(N0/2)    
    '''
    noise = numpy_to_torch_tensor(np.random.normal(0, np.sqrt(N0 / 2), [n,2])).float()
    return noise

def get_N0(SNR_db, signal_power):
    '''
    SNR_db: The desired signal to noise ratio in db scale
    signal_power: The signal power in linear scale
    '''
    SNR = 10**(0.1*SNR_db) #Get SNR in linear scale
    N0 = signal_power/SNR
    return N0

def add_cartesian_awgn(data_c, SNR_db, signal_power = 1.0):
    '''
    Inputs:
    data_c: torch.tensor of type float and shape (n,2) containing modulated symbols
    SNR_db: Desired signal to noise ratio in db
    signal_power: Signal power in linear scale (Default = 1.0)
    Output:
    data_c_noisy: Noisy modulated symbols where noise such that we get desired SNR_db
    '''
    
    N0 = get_N0(SNR_db=SNR_db, signal_power=signal_power)
    # print(N0)
    noise = get_awgn(N0=N0, n = data_c.shape[0])   
    data_c_noisy = data_c + noise
    return data_c_noisy

def add_complex_awgn(data_c, SNR_db, signal_power = 1.0):
    '''
    Inputs:
    data_c: numpy.array of type complex and shape (n,1) containing modulated symbols
    SNR_db: Desired signal to noise ratio in db
    signal_power: Signal power in linear scale (Default = 1.0)
    Output:
    data_c_noisy: Noisy modulated symbols where noise such that we get desired SNR_db
    '''
    
    N0 = get_N0(SNR_db=SNR_db, signal_power=signal_power)
    noise = torch_tensor_to_numpy(get_awgn(N0=N0, n = data_c.shape[0]))   
    data_c_noisy = data_c + noise[:,0] + 1j*noise[:,1]
    return data_c_noisy

def get_test_SNR_dbs():
    test_SNR_dbs ={1: {'ber': [11.399999999999924, 9.59999999999993, 8.399999999999935, 6.79999999999994, 4.399999999999949, -0.8000000000000327], 'ber_roundtrip': [11.799999999999923, 9.999999999999929, 8.799999999999933, 7.399999999999938, 5.399999999999945, 1.1999999999999602], 'ser': [11.399999999999924, 9.59999999999993, 8.399999999999935, 6.79999999999994, 4.399999999999949, -0.8000000000000327], 'ser_roundtrip': [11.799999999999923, 9.999999999999929, 8.799999999999933, 7.399999999999938, 5.399999999999945, 1.1999999999999602]}, 2: {'ber': [14.599999999999913, 12.59999999999992, 11.399999999999924, 9.79999999999993, 7.399999999999938, 2.1999999999999567], 'ber_roundtrip': [14.599999999999913, 12.999999999999918, 11.999999999999922, 10.399999999999928, 8.399999999999935, 4.1999999999999496], 'ser': [14.599999999999913, 12.999999999999918, 11.999999999999922, 10.399999999999928, 8.399999999999935, 4.399999999999949], 'ser_roundtrip': [14.599999999999913, 13.199999999999918, 12.199999999999921, 10.999999999999925, 8.999999999999932, 5.799999999999944]}, 3: {'ber': [19.799999999999894, 17.7999999999999, 16.599999999999905, 14.799999999999912, 12.199999999999921, 5.999999999999943], 'ber_roundtrip': [19.999999999999893, 18.1999999999999, 16.999999999999904, 15.39999999999991, 13.199999999999918, 8.399999999999935], 'ser': [19.799999999999894, 18.3999999999999, 17.199999999999903, 15.799999999999908, 13.599999999999916, 9.79999999999993], 'ser_roundtrip': [19.999999999999893, 18.5999999999999, 17.599999999999902, 16.199999999999907, 14.399999999999913, 11.199999999999925]}, 4: {'ber': [21.799999999999887, 19.599999999999895, 18.3999999999999, 16.599999999999905, 13.999999999999915, 7.999999999999936], 'ber_roundtrip': [21.799999999999887, 19.999999999999893, 18.799999999999898, 17.199999999999903, 14.999999999999911, 10.399999999999928], 'ser': [21.799999999999887, 20.199999999999893, 19.199999999999896, 17.7999999999999, 15.799999999999908, 12.199999999999921], 'ser_roundtrip': [21.799999999999887, 20.399999999999892, 19.399999999999896, 18.1999999999999, 16.399999999999906, 13.599999999999916]}, 6: {'ber': [27.999999999999865, 25.599999999999874, 24.399999999999878, 22.59999999999988, 19.799999999999894, 12.999999999999918], 'ber_roundtrip': [28.19999999999986, 25.999999999999872, 24.799999999999876, 23.199999999999882, 20.79999999999989, 15.599999999999909], 'ser': [27.999999999999865, 26.39999999999987, 25.399999999999878, 23.99999999999988, 22.19999999999989, 18.799999999999898], 'ser_roundtrip': [28.19999999999986, 26.79999999999987, 25.79999999999987, 24.59999999999988, 22.799999999999883, 19.999999999999893]}}
    
    return test_SNR_dbs
##############################################################################
# Measuring effect of noise and performance
##############################################################################
def get_grid_2d(grid = [-1.5,1.5], points_per_dim = 100):
    grid_1d = np.linspace(grid[0], grid[1], points_per_dim) 
    grid_2d = np.squeeze(np.array(list(itertools.product(grid_1d, grid_1d))))
    return numpy_to_torch_tensor(grid_2d).float()
    
def test_empirical_SNR_db(data_c, data_c_noisy):
    noise = data_c_noisy - data_c
    emp_signal_power = torch.mean(torch.sum(data_c**2, -1))
    emp_signal_power_db = 10*torch.log10(emp_signal_power)
    print("emp_signal_power_db: ", emp_signal_power_db)
    
    emp_noise_power = torch.mean(torch.sum(noise**2, -1))
    emp_noise_power_db = 10*torch.log10(emp_noise_power)
    print("Empiricial noise power db: ", emp_noise_power_db)

    emp_SNR_db = emp_signal_power_db - emp_noise_power_db
    print("Empirical SNR db: ", emp_SNR_db)
    
def get_symbol_error_rate(data_si, labels_si_g):
    '''
    data_si: torch.tensor of shape [n,1]
    labels_si_g: torch.tensor of shape [n,1]
    Returns the number of indices where these differ divided by n
    '''
    if len(data_si.shape) == 2:
        data_si = data_si[:,0]
    if len(labels_si_g.shape) == 2:
        labels_si_g = labels_si_g[:,0]
    return labels_si_g[labels_si_g!=data_si].shape[0]/labels_si_g.shape[0]

def get_bit_error_rate(data_si, labels_si_g, bits_per_symbol):
    '''
    data_si: torch.tensor OR numpy.array of shape [n,1]
    labels_si_g: torch.tensor OR numpy.array of shape [n,1]
    bits_per_symbol: integer corresponding to number of bits per symbol
    Returns the number of bit errors divided by n
    '''
    error_values = np.array(
        [bin(x).count('1') for x in range(2 ** bits_per_symbol)]) # Compute bit errors corresponding to each integer 
    if isinstance(data_si, torch.Tensor):
        error_values = numpy_to_torch_tensor(error_values).long() 

    diff = data_si ^ labels_si_g  # xor to find differences in two streams
    bit_errors = error_values[diff]
    if isinstance(data_si, torch.Tensor):
        bit_error_rate = torch.mean(bit_errors.float())/bits_per_symbol
        bit_error_rate = torch_tensor_to_numpy(bit_error_rate)
    elif isinstance(data_si, np.ndarray):
        bit_error_rate = np.mean(bit_errors)/bits_per_symbol
    return bit_error_rate

##############################################################################
# Data Generation
##############################################################################

def get_random_bits(n):
    '''Return np integer array of 0-1 of shape [n]'''
    return np.random.randint(low=0, high=2, size=[n])

def get_random_data_si(n, bits_per_symbol):
    '''
    Generate random data for integer representation of symbols between [0, 2**bits_per_symbol]
    shape [n] --> n random symbols = n*bits_per_symbol random bits
    '''
    return np.random.randint(low=0,high=2**bits_per_symbol, size=[n])

# def add_cartesian_awgn(data_c, bits_per_symbol, EbN0_db, Es=1):
#     '''
#     Add AWGN corresponding to EbN0 value in db with Eb value in linear scale.
#     Inputs:
#     data_c : [n,2] torch tensor OR np.array of the complex modulated signal in the cartesian coordinate form
#     '''
#     #First get N0 
#     Eb = Es/bits_per_symbol
#     EbN0 = 10**(0.1*EbN0_db)
#     N0 = Eb/(EbN0)    
#     #Add noise corresponding to N0
#     noise = np.random.normal(0, np.sqrt(N0 / 2), data_c.shape)
#     if isinstance(data_c, torch.Tensor):
#         noise = numpy_to_torch_tensor(noise)
#     return data_c + noise  

# def add_complex_awgn(data_c, N0):
#     '''
#     Adds AWGN (complex noise) of specified N0 values
#     Inputs:
#     data: np.array of complex type of shape [n]
#     N0: Noise power value can be of the following types:
#     1) float
#     2) list of length m
#     3) np.array of shape [m]
#     Depending on type of N0, the output is:
#     1) (if N0 is of type float) noisy_data: np.array of shape [n]
#     2) (if N0 is list or np.array) noisy_data_list: list of np.array of len(m) with each element of shape [n]
#     '''
#     if not isinstance(data_c, np.ndarray):
#         raise TypeError("add_complex_awgn: data_c must be an np_array") from error
#     # Convert to array
#     N0_array = np.reshape(np.array(N0), [-1])
#     if N0_array.shape[0] == 1:  # Means only one value
#         N0 = N0_array[0]
#         noise_re = np.random.normal(0, np.sqrt(N0 / 2), data_c.shape)
#         noise_im = np.random.normal(0, np.sqrt(N0 / 2), data_c.shape)
#         noisy_data = data_c + (noise_re + 1j * noise_im)
#         return noisy_data
#     else:
#         noisy_data_list = []
#         for k in range(N0_array.shape[0]):
#             N0 = N0_array[k]
#             noise_re = np.random.normal(0, np.sqrt(N0 / 2), data_c.shape)
#             noise_im = np.random.normal(0, np.sqrt(N0 / 2), data_c.shape)
#             cur_noisy_data = data_c + (noise_re + 1j * noise_im)
#             noisy_data_list.append(cur_noisy_data)
#         return noisy_data_list


# def get_BER(labels_si, labels_si_g, bits_per_symbol):
#     '''
#     Evaluates bit errors given integers corresponding to true and estimated symbols
#     Inputs:
#     labels_si: np.array of type integer of shape [m] containing true labels for points
#     labels_si_g: np.array of type integer of shape [m] containing estimated labels for points
#     Output:
#     BER: Number of bits in error divided by total number of bits
#     '''
#     error_values = np.array(
#         [bin(x).count('1') for x in range(2 ** bits_per_symbol)])  # Compute bit errors corresponding
#     # to each integer
#     diff = labels_si ^ labels_si_g  # xor to find differences in two streams
#     bit_errors = np.sum(error_values[diff])
#     BER = bit_errors / (labels_si.shape[0] * bits_per_symbol)
#     return BER

# def get_symbol_error_rate(data_si, labels_si_g):
#     '''
#     Evaluates symbol errors given integers corresponding to true and estimated symbols
#     data_si: torch.tensor/np.array of shape [N] containing guesses of symbols
#     labels_si_g: torch.tensor/np.array of shape [N] containing guesses of symbols
#     Returns the number of indices where these differ divided by n
#     '''
#     SER = labels_si_g[labels_si_g != data_si].shape[0]/labels_si_g.shape[0]
#     return SER


def get_noisy_modulated_data(num_symbols, mod_types, EbN0s):
    '''
    Inputs:
    num_symbols: scalar number of symbols
    mod_types: np.array or list of shape [m]
    EbN0s: np.array or list of shape [r]
    Output:
    noisy_data_c_dict: Dictionary with keys (mod_type, EbN0) pair and values complex stream corresponding to noisy
                 modulated signals
    labels_si_dict: Dictionary with keys (mod_type, EbN0) pair and values integers corresponding to true labels. If only one element in dicitonary return just the values
    '''
    from modulators.modulator_classic import ModulatorClassic
    noisy_data_c_dict = {}
    labels_si_dict = {}
    mod_types = np.reshape(np.array(mod_types), [-1])
    EbN0s = np.reshape(np.array(EbN0s), [-1])
    for mod_type in list(mod_types):
        mapper_mod = ModulatorClassic(mod_type=mod_type)
        data_si = np.random.randint(low=0, high=2 ** mapper_mod.bits_per_symbol, size=[num_symbols])
        labels_si = data_si
        data_c = mapper_mod.modulate(data_si=data_si)
        N0_array = mapper_mod.get_N0(EbN0=EbN0s)
        noisy_data_c_list = add_awgn(data_c=data_c, N0=N0_array)
        if len(list(EbN0s)) == 1:
            noisy_data_c_list = [noisy_data_c_list]
        for k, EbN0 in enumerate(list(EbN0s)):
            key = (mod_type, EbN0)
            noisy_data_c = noisy_data_c_list[k]
            noisy_data_c_dict[key] = noisy_data_c
            labels_si_dict[key] = labels_si

    #     if len(noisy_data_c_dict) is 1:
    #         cur_key = (mod_types[0],EbN0s[0])
    #         return noisy_data_c_dict[cur_key], labels_si_dict[cur_key]
    #     else:
    return noisy_data_c_dict, labels_si_dict


##############################################################################
# Data Manipulation
##############################################################################

def int_to_base(x, base, size=None, order='decreasing'):
    '''
    Convert integer sequence to sequence of sequence of integers in given base
    Inputs:
    x: np array of shape [n]
    base: scalar, the base to convert to
    size: length of representation. if None use minimum possible length required to represent in given base
    order: If decreasing MSB comes first
    '''
    x = np.asarray(x)
    if size is None:
        size = int(np.ceil(np.log(np.max(x)) / np.log(base)))
    if order == "decreasing":
        powers = base ** np.arange(size - 1, -1, -1)
    else:
        powers = base ** np.arange(size)
    digits = (x.reshape(x.shape + (1,)) // powers) % base
    return digits


def bits_to_symbols(data_b, bits_per_symbol):
    '''
    Converts array of bits into array of bit representation of symbol
    Inputs:
    data_b: np.array of type integer containing 0-1 entries of shape [n]
    bits_per_symbol: scalar such that n is divisible by bits_per_symbol
    Output:
    data_sb: np.array of type integer containing 0-1 entries of shape [m=n/bits_per_symbol, bits_per_symbol]
    '''
    if data_b.shape[0] % bits_per_symbol == 0:
        data_sb = np.reshape(data_b, [-1, bits_per_symbol])
        return data_sb
    else:
        raise ValueError('Length of data_b: ' + str(data_b.shape[0]) + ' is not divisible by bits_per_symbol: ' + str(
            bits_per_symbol))


def symbols_to_integers(data_sb):
    '''
    Converts array of bit represented symbols into integer represented symbols
    Inputs:
    data_sb: np.array of type integer containing 0-1 entries of shape [m, bits_per_symbol]       
    Output:
    data_si: np.array of type integer containing integer representation of rows of data_sb, 
             of shape [m]
    '''
    data_si = data_sb[:, 0]
    for i in range(1, data_sb.shape[1]):
        data_si = (data_si << 1) + data_sb[:, i]
    return data_si


def bits_to_integers(data_b, bits_per_symbol):
    '''
    Converts array of bits into array of integer representation of symbol
    Inputs:
    data_b: np.array of type integer containing 0-1 entries of shape [n]
    bits_per_symbol: scalar such that n is divisible by bits_per_symbol
    Output:
    data_si: np.array of type integer containing integer representation of rows of data_sb, 
             of shape [m=n/bits_per_symbol]
    '''
    data_sb = bits_to_symbols(data_b=data_b, bits_per_symbol=bits_per_symbol)
    data_si = symbols_to_integers(data_sb=data_sb)
    return data_si


def integers_to_symbols(data_si, bits_per_symbol):
    '''
    Converts array of integer representation of bits to bit symbol representation
    Inputs:
    data_si: np.array of type integer containing integer representation of rows of data_sb, 
             of shape [m]
    bits_per_symbol: scalar
    Output:
    data_sb: np.array of type integer containing 0-1 entries of shape [m, bits_per_symbol]    
    '''

    data_sb = np.zeros((data_si.shape[0], bits_per_symbol), dtype='int')
    for k in range(bits_per_symbol - 1, -1, -1):
        data_sb[:, k] = data_si - ((data_si >> 1) << 1)
        data_si = data_si >> 1
    return data_sb


def symbols_to_bits(data_sb):
    '''
    Converts array of bit representation of symbols to array of bits
    Inputs:
    data_sb: np.array of type integer containing 0-1 entries of shape [m, bits_per_symbol] 
    Output:
    data_b: np.array of type integer containing 0-1 entries of shape [n=m*bits_per_symbol]
    '''
    data_b = np.reshape(data_sb, [-1])
    return data_b


def integers_to_bits(data_si, bits_per_symbol):
    '''
    Converts array of integer representation of bits to an array of bits
    Inputs:
    data_si: np.array of type integer containing integer representation of rows of data_sb, 
             of shape [m]
    bits_per_symbol: scalar
    Output:
    data_b: np.array of type integer containing 0-1 entries of shape [n=m*bits_per_symbol]  
    '''
    data_sb = integers_to_symbols(data_si=data_si, bits_per_symbol=bits_per_symbol)
    data_b = symbols_to_bits(data_sb=data_sb)
    return data_b


def complex_to_cartesian_2d(data_c):
    '''
    Converts complex numbers to 2D cartesian representation
    Inputs:
    data_c: np.array of type complex of shape [N]
    Output:
    data_d: np.array of type float of shape [N,2]    
    '''
    data_d = np.transpose(np.vstack([data_c.real, data_c.imag]))
    return data_d

def cartesian_2d_to_complex(data_d):
    '''
    Converts 2D cartesian representation to complex numbers
    Inputs:
    data_c: np.array of type float of shape [N,2]
    Output:
    data_d: np.array of type complex of shape [N]    
    '''
    data_c = data_d.astype(np.complex64)
    data_c[:,1] *= 1j
    data_c = np.sum(data_c, axis=1)
    return data_c

def get_all_unique_symbols(bits_per_symbol):
    '''
    Returns np.array of shape [2**bits_per_symbol, bits_per_symbol] containing bit representaion of symbols   
    '''
    data_si = np.arange(2**bits_per_symbol)
    return integers_to_symbols(data_si=data_si, bits_per_symbol=bits_per_symbol)

def torch_tensor_to_numpy(x, dtype=np.float32):
    #Convert pytorch tensor to numpy array
    if x is not None:
        return x.data.numpy().astype(dtype)

def numpy_to_torch_tensor(x, dtype=torch.float32):
    #Convert numpy array to pytorch tensor
    if x is not None:
        return torch.from_numpy(x).type(dtype)

def rotate_clockwise(vector, rotate_angle):
    '''
    vector: torch.tensor of shape [n,2]
    rotate_angle: angle in rads to rotate clockwise by
    '''    
    rotation_matrix = torch.tensor([[np.cos(-rotate_angle), np.sin(-rotate_angle)],[np.sin(rotate_angle), np.cos(-rotate_angle)]])
    return torch.mm(vector, rotation_matrix)

################################################################################
# Loss functions
################################################################################

def get_bit_l1_loss(labels_si_g, labels_si, bits_per_symbol):
    '''
    Inputs:
        labels_si_g: np.array of type integer and shape [N] containing guesses of symbols
        labels_si: np.array of type integer and shape [N] containing true symbol
    Outputs:
        loss: np.array of type float and shape [N] containing bit errors for each symbols            
    '''
    # Compute bit errors corresponding to each integer
    error_values = np.array([bin(x).count('1') for x in range(2 ** bits_per_symbol)])
    diff = labels_si ^ labels_si_g  # xor to find differences in two streams
    loss = error_values[diff]
    return loss  # Only if you get all bits right do you get a positive reward


def get_cross_agent_loss(labels_a1, labels_a2, method='robust', return_translation=False):
    '''
    Inputs:
        labels_a1: np.array of type integer and shape [N] containing agent 1's guesses of symbols
        labels_a2: np.array of type integer and shape [N] containing agent 2's guesses of symbols (same as labels_a1, but may have different "representation")
        method: 'simple' (default) -- label matching done by first-seen
                'robust'           -- label matching done by maximizing overlap
        return_translation: True to return mapping between agent 1 and agent 2 symbols used to compute loss
    Outputs:
        loss: np.array of type integer and shape [N] and containing 1 if corresponding symbol matches across agents and 0 if it not
        translation: list of 2 dictionaries
                        1) key = label from agent 1, value is translation to agent 2
                        2) key = label from agent 2, value is transaltion to agent 1
    '''
    for labels in [labels_a1, labels_a2]:
        translate = {}
        counter = 0
        for i in range(len(labels)):
            l = labels[i]
            if l not in translate:
                translate[l] = counter
                counter += 1
            labels[i] = translate[l]

    # uses a greedy optimization approach (matching pursuit)
    # to find the label pairing that minimizes loss
    if method == 'robust':
        n = max(np.max(labels_a1), np.max(labels_a2))
        a1_dict = {i: (labels_a1 == i).astype(int) for i in range(n + 1)}
        a2_dict = {i: (labels_a2 == i).astype(int) for i in range(n + 1)}
        for i in range(n + 1):
            a1_elem = a1_dict[i]
            best = [len(a1_elem) + 1, -1]
            for k in a2_dict.keys():
                score = np.linalg.norm(a2_dict[k] - a1_elem, ord=1)
                if score < best[0]:
                    best[0] = score
                    best[1] = k
            labels_a1[np.where(a1_elem)] = best[1]
            a2_dict.pop(best[1])
    return (labels_a1 == labels_a2).astype(int)


def get_cross_agent_ber(labels_a1, labels_a2, bits_per_symbol, method='robust'):
    '''
    Inputs:
        labels_a1: np.array of type integer and shape [N] containing agent 1's guesses of symbols
        labels_a2: np.array of type integer and shape [N] containing agent 2's guesses of symbols (same as labels_a1, but may have different "representation")
        method: 'simple' (default) -- label matching done by first-seen
                'robust'           -- label matching done by maximizing overlap
    Outputs:
        loss: np.array of type integer and shape [N] and containing 1 if corresponding symbol matches across agents and 0 if it not
        translation: list of 2 dictionaries
                        1) key = label from agent 1, value is translation to agent 2
                        2) key = label from agent 2, value is transaltion to agent 1
    '''
    for labels in [labels_a1, labels_a2]:
        translate = {}
        counter = 0
        for i in range(len(labels)):
            l = labels[i]
            if l not in translate:
                translate[l] = counter
                counter += 1
            labels[i] = translate[l]

    # uses a greedy optimization approach (matching pursuit)
    # to find the label pairing that minimizes loss
    if method == 'robust':
        n = max(np.max(labels_a1), np.max(labels_a2))
        a1_dict = {i: (labels_a1 == i).astype(int) for i in range(n + 1)}
        a2_dict = {i: (labels_a2 == i).astype(int) for i in range(n + 1)}
        for i in range(n + 1):
            a1_elem = a1_dict[i]
            best = [len(a1_elem), -1]
            for k in a2_dict.keys():
                score = np.linalg.norm(a2_dict[k] - a1_elem, ord=1)
                if score < best[0]:
                    best[0] = score
                    best[1] = k
            labels_a1[np.where(a1_elem)] = best[1]
            a2_dict.pop(best[1])

    error_values = np.array(
        [bin(x).count('1') for x in range(2 ** bits_per_symbol)])  # Compute bit errors corresponding
    # to each integer
    diff = labels_a1 ^ labels_a2  # xor to find differences in two streams
    bit_errors = np.sum(error_values[diff])
    BER = bit_errors / (labels_a1.shape[0] * bits_per_symbol)
    return BER


def get_complex_l2_loss(data_c, data_c_g):
    '''
    Policy update function. Calls self.update_op.

    Inputs:
        data_c_g: np.array of type complex and shape [N] containing guesses of modulated symbols
        data_c: np.array of type complex and shape [N] containing true modulated symbols
    Outputs:
        loss: np.array of type float and shape [N] containing l2 error for each symbols            
    '''
    # Compute bit errors corresponding to each integer
    loss = np.abs(data_c - data_c_g)
    return loss


def cluster_kmeans(data, k, num_iterations, ref_means=None):
    '''
    Clusters data into k classes and assigns labels based on closest clusters from ref_means
    Inputs:
        ref_means: the means of clusters indexed by 0,...,k-1 that we want to try and match
        k: Number of clusters
        num_iterations: Number of iterations to run the kmeans clustering algorithm
    Outputs:
        assign: np.array of type integer and shape [N] in which each element is the cluster (index of the mean)\
        assigned to the respective symbol
        means: np.array of type complex and shape [k] containing means of each cluster
    '''
    mapper_kmeans = Kmeans(k=k)
    mapper_kmeans.initialize(data=data, hard=True)
    assign = mapper_kmeans.iterate(data=data, num_iterations=num_iterations)
    means = mapper_kmeans.means
    new_means = np.array(means, copy=True)
    new_assign = np.array(assign, copy=True)
    if ref_means is not None:
        # Find distance between each ref mean to the newly found means
        dist = np.abs(ref_means[:, None] - means[None,
                                           :])  # Element dist[i,j] contains distance between ref_means[i] and means[j]
        # Find minimum distance between 2 clusters say dist[i0,j0] and then assign new_means[i0]
        # as means[j0] and new_labels[j0] as i0
        for i in range(k):
            i0, j0 = np.unravel_index(np.argmin(dist, axis=None), dims=dist.shape)
            new_means[i0] = means[j0]
            dist[:, j0] = float('inf')
            dist[i0, :] = float('inf')
            new_assign[assign == j0] = i0

    return new_assign, new_means


def get_cluster_loss(data_c, data_c_g, k, num_iterations=20, hard=False):
    '''
        Returns loss based on clustered versions of data_c, data_c_g for each point as distance between
        respective cluster centers when 'stability' maintained using ref_means ideas (look at kmeans class)               
        Inputs:
        data_c_g: np.array of type complex and shape [N] containing guesses of modulated symbols
        data_c: np.array of type complex and shape [N] containing true modulated symbols
        k: Number of clusters
        num_iterations: Number of iterations to run the kmeans clustering algorithm
        hard: (Default true) If true 0-1 loss based on if same cluster, else loss based on distane between cluster centers
        Outputs:
        loss: np.array of type float and shape [N] containing l2 error for each symbols  
        '''

    assign, means = cluster_kmeans(data=data_c, k=k, num_iterations=num_iterations)
    assign_g, means_g = cluster_kmeans(data=data_c_g, k=k, num_iterations=num_iterations, ref_means=means)
    if hard is True:
        loss = np.array(assign != assign_g, dtype='float')
    else:
        loss = np.abs(means[assign] - means_g[assign_g])
    return loss


# ##################################################################################
# # K means clustering
# ###################################################################################

################################################################################
# Code for testing functions
################################################################################
def main():
    # Parameters
    np.random.seed(7)
    num_symbols = 5
    bits_per_symbol = 4

    # Test bit generation
    data_b = get_random_bits(n=num_symbols * bits_per_symbol)
    print("data_b")
    print(data_b)

    # Test bits to symbols conversion
    data_sb = bits_to_symbols(data_b=data_b, bits_per_symbol=bits_per_symbol)
    print("data_sb")
    print(data_sb)

    # Test symbols to integers conversion
    data_si = symbols_to_integers(data_sb=data_sb)
    print("data_si")
    print(data_si)

    # Test bits to integers conversion
    data_si_2 = bits_to_integers(data_b=data_b, bits_per_symbol=bits_per_symbol)
    print("data_si_2")
    print(data_si_2)

    # Test integers to symbols conversion
    data_sb_2 = integers_to_symbols(data_si=data_si_2, bits_per_symbol=bits_per_symbol)
    print("data_sb_2")
    print(data_sb_2)

    # Test symbols to bits conversion
    data_b_2 = symbols_to_bits(data_sb=data_sb_2)
    print("data_b_2")
    print(data_b_2)

    # Test integers to bits conversion
    data_b_3 = integers_to_bits(data_si=data_si_2, bits_per_symbol=bits_per_symbol)
    print("data_b_3")
    print(data_b_3)


if __name__ == '__main__':
    main()
