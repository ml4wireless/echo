
# coding: utf-8

# In[3]:


#Contains functions for classic modulation schemes
import numpy as np
from scipy import cos, sin, pi
##########################################################
# Helper functions for symbol to complex number mapping
##########################################################
def get_gray_code(n):
    '''
    Returns inverse gray code map for numbers upto 2**n-1
    Inputs:
    n: Number of bits. Find inverse gray code maps for [0,2**n-1]
    Outputs:
    g_inverse: np.array of type integer of shape [2**n]. For i in [0,2**n-1], \
    g_inverse[i] contains position where i goes to when gray coded
    '''
    if n < 1:
        g = []
    else:
        g = ['0', '1']
        n -= 1
        while n > 0:
            k = len(g)
            for i in range(k-1, -1, -1):
                char = '1' + g[i]
                g.append(char)
            for i in range(k-1, -1, -1):
                g[i] = '0' + g[i]
            n -= 1
    g_inverse = np.zeros((len(g),), dtype = 'int')
    for i,gi in enumerate(g):
        g_inverse[int(gi,2)] = int(i)
    return g_inverse


def get_mpsk_symbol(k, M, gray_code, offset=0):
    '''
    Returns complex number corresponding to k (between 0 and M-1) for MPSK
    Inputs:
    k: Integer in [0,M-1] 
    M: Integer (2 for BPSK, 4 for QPSK, 8 for 8PSK)
    gray_code: Inverse gray code map for n=log2(M)  
    
    Output:
    mpsk_symbol: Complex number representing I + jQ for the symbol k in MPSK scheme having unit amplitude
    '''
    mpsk_symbol = 0
    if k < M:
        k_gray = gray_code[k]
        mpsk_symbol = cos(2*pi*k_gray/M + offset) + 1j*sin(2*pi*k_gray/M + offset)            
    else:
        raise ValueError("Error: k: " + str(k)+ " exceeds M-1: " + str(M-1))        
    return mpsk_symbol 

def gray_coded_IQ(k,d,gray_code):
    '''
    Splits bit representation of k into two halves and returns\
    gray coded I and Q to be used to perform the QAM modulation
    Inputs:
    k: Integer
    d: Integer, Length of bit representation
    gray_code: Inverse gray code map for n = d/2
    Outputs:
    kI_gray: Integer in 0,2**(d/2) - 1. Gray coded integer for I
    kQ_gray: Integer in 0,2**(d/2) - 1. Gray coded integer for Q
    '''
    IQ = np.binary_repr(k,d)    
    I = IQ[:d//2]
    Q = IQ[d//2:]   
    kI = int(I,2)
    kQ = int(Q,2)    
    kI_gray = gray_code[kI]
    kQ_gray = gray_code[kQ]    
    return kI_gray, kQ_gray

def get_mqam_symbol(k, M, gray_code): #M must be a square number
    '''
    Returns complex number corresponding to k (between 0 and M-1) for MQAM
    Inputs:
    k: Integer in [0,M-1] 
    M: Integer (Must be perfect square) (16 for QAM16, 64 for QAM64)
    gray_code: Inverse gray code map for n=log2(sqrt(M))   
    
    Output:
    mpsk_symbol: Complex number representing I + jQ for the symbol k in MPSK scheme
    '''
    mqam_symbol = 0
    if k < M:        
        K = np.sqrt(M)
        #Break into I and Q
        d = int(np.log2(M))
        kI_gray, kQ_gray = gray_coded_IQ(k,d, gray_code)        
        scaling_factor = (1/(np.sqrt((2.0/3)*(M-1)))) #scaling factor so overall constellation has unit average energy
        mqam_symbol_I = scaling_factor*((-K+1) + (2*kI_gray))
        mqam_symbol_Q = scaling_factor*((-K+1) + (2*kQ_gray))        
        mqam_symbol = mqam_symbol_I + 1j*mqam_symbol_Q
        
    else:
        raise ValueError("Error: k: " + str(k)+ " exceeds M-1: " + str(M-1))        
    return mqam_symbol

def get_mpsk_modulation_maps(M):   
    '''
    Returns complex numbers and legends corresponding to given MPSK scheme
    Inputs:
    M: scalar integer, Number of points in constellation (eg. M = 2 for BPSK, M = 4 for QPSK)
    Outputs:
    symbol_map: Dict with keys [0,M-1] and values complex numbers corresponding to I + jQ (with amplitude 1)
    legend_amp: Dict with keys [0,M-1] and values string labels corresponding to their bit representaion
    '''
    #Get number of bits d from M
    d = int(np.log2(M))    
    #Fix offset
    if M == 2:
        offset = 0 #No offset for BPSK
    else:
        offset = pi/M    
    #Get gray code map for d
    gray_code = get_gray_code(n=d)    
    #Initialize outputls
    symbol_map = {}
    legend_map = {}
    for i in range(M):        
        #Get symbol
        symbol_map[i] = get_mpsk_symbol(k=i,M=M,gray_code=gray_code,offset=offset)        
        #Get legend  
        i_binary = np.binary_repr(i,width=d)
        symbol = str(i_binary)
        legend_map[i] = str(symbol)           
    return symbol_map, legend_map

def get_mqam_modulation_maps(M):   
    '''
    Returns complex numbers and legends corresponding to given MQAM scheme
    Inputs:
    M: scalar integer perfect square, Number of points in constellation (eg. 16 for QAM16, 64 for QAM64)
    Outputs:
    symbol_map: Dict with keys [0,M-1] and values complex numbers corresponding to I + jQ. Constellation 
                points are normalized to have unit energy
    legend_amp: Dict with keys [0,M-1] and values string labels corresponding to their bit representaion
    '''
    #Get number of bits d from M
    d = int(np.log2(M))    
    #Fix offset
    if M == 2:
        offset = 0 #No offset for BPSK
    else:
        offset = pi/M    
    #Get gray code map for d
    gray_code = get_gray_code(n=d)    
    #Initialize outputls
    symbol_map = {}
    legend_map = {}
    for i in range(M):        
        #Get symbol
        symbol_map[i] = get_mqam_symbol(k=i,M=M,gray_code=gray_code)        
        #Get legend  
        i_binary = np.binary_repr(i,width=d)
        symbol = str(i_binary)
        legend_map[i] = str(symbol)           
    return symbol_map, legend_map


def get_modulation_maps(mod_type):     
    '''
    Wrapper function for getting modulation types
    Inputs:
    mod_type: Modulation type (M is determined from mod_type internally)
    Outputs:
    symbol_map: Dict with keys [0,M-1] and values complex numbers corresponding to I + jQ. Constellation 
                points are normalized to have unit energy
    legend_amp: Dict with keys [0,M-1] and values string labels corresponding to their bit representaion
    
    '''
    #Dictionary for mapping between mod_type and M
    mod_type_to_M = {'BPSK':2, 'QPSK':4, '8PSK':8, 'QAM16':16, 'QAM64':64}
    psk_mod_types = ['BPSK', 'QPSK', '8PSK'] #List of MPSK schemese
    qam_mod_types = ['QAM16', 'QAM64'] #List of MQAM schemese
    mod_type_to_bits_per_symbol = {'BPSK':1, 'QPSK':2, '8PSK':3, 'QAM16':4, 'QAM64':6}
    if mod_type  in psk_mod_types:
        symbol_map, legend_map = get_mpsk_modulation_maps(M=mod_type_to_M[mod_type])
    elif mod_type in qam_mod_types:
        symbol_map, legend_map = get_mqam_modulation_maps(M=mod_type_to_M[mod_type])
    else:
        raise ValueError('Unknown mod_type: ' + str(mod_type) + '. mod_type must be in ' + str(list(mod_type_to_M.keys())))
    return mod_type_to_bits_per_symbol[mod_type], symbol_map, legend_map


symbol_maps = {}
#For BPSK
symbol_maps[1] = np.array([[1.0,0.0],[-1.0,0.0]])
#For QPSK
symbol_maps[2] =  np.array([[0.7071067811865476,0.7071067811865475],[-0.7071067811865475,0.7071067811865476],[0.7071067811865474,-0.7071067811865477],[-0.7071067811865477,-0.7071067811865475]])
#For 8PSK
symbol_maps[3] = np.array([[0.9238795325112867,0.3826834323650898],[0.38268343236508984,0.9238795325112867],[-0.9238795325112867,0.3826834323650899],[-0.3826834323650897,0.9238795325112867],[0.9238795325112865,-0.3826834323650904],[0.38268343236509,-0.9238795325112866],[-0.9238795325112868,-0.38268343236508967],[-0.38268343236509034,-0.9238795325112865]])
#For 16QAM
symbol_maps[4] = np.array([[-0.9486832980505138,-0.9486832980505138],[-0.9486832980505138,-0.31622776601683794],[-0.9486832980505138,0.9486832980505138],[-0.9486832980505138,0.31622776601683794],[-0.31622776601683794,-0.9486832980505138],[-0.31622776601683794,-0.31622776601683794],[-0.31622776601683794,0.9486832980505138],[-0.31622776601683794,0.31622776601683794],[0.9486832980505138,-0.9486832980505138],[0.9486832980505138,-0.31622776601683794],[0.9486832980505138,0.9486832980505138],[0.9486832980505138,0.31622776601683794],[0.31622776601683794,-0.9486832980505138],[0.31622776601683794,-0.31622776601683794],[0.31622776601683794,0.9486832980505138],[0.31622776601683794,0.31622776601683794]])
#For 64QAM
symbol_maps[6] = np.array([[-1.0801234497346435,-1.0801234497346435],[-1.0801234497346435,-0.7715167498104596],[-1.0801234497346435,-0.1543033499620919],[-1.0801234497346435,-0.4629100498862757],[-1.0801234497346435,1.0801234497346435],[-1.0801234497346435,0.7715167498104596],[-1.0801234497346435,0.1543033499620919],[-1.0801234497346435,0.4629100498862757],[-0.7715167498104596,-1.0801234497346435],[-0.7715167498104596,-0.7715167498104596],[-0.7715167498104596,-0.1543033499620919],[-0.7715167498104596,-0.4629100498862757],[-0.7715167498104596,1.0801234497346435],[-0.7715167498104596,0.7715167498104596],[-0.7715167498104596,0.1543033499620919],[-0.7715167498104596,0.4629100498862757],[-0.1543033499620919,-1.0801234497346435],[-0.1543033499620919,-0.7715167498104596],[-0.1543033499620919,-0.1543033499620919],[-0.1543033499620919,-0.4629100498862757],[-0.1543033499620919,1.0801234497346435],[-0.1543033499620919,0.7715167498104596],[-0.1543033499620919,0.1543033499620919],[-0.1543033499620919,0.4629100498862757],[-0.4629100498862757,-1.0801234497346435],[-0.4629100498862757,-0.7715167498104596],[-0.4629100498862757,-0.1543033499620919],[-0.4629100498862757,-0.4629100498862757],[-0.4629100498862757,1.0801234497346435],[-0.4629100498862757,0.7715167498104596],[-0.4629100498862757,0.1543033499620919],[-0.4629100498862757,0.4629100498862757],[1.0801234497346435,-1.0801234497346435],[1.0801234497346435,-0.7715167498104596],[1.0801234497346435,-0.1543033499620919],[1.0801234497346435,-0.4629100498862757],[1.0801234497346435,1.0801234497346435],[1.0801234497346435,0.7715167498104596],[1.0801234497346435,0.1543033499620919],[1.0801234497346435,0.4629100498862757],[0.7715167498104596,-1.0801234497346435],[0.7715167498104596,-0.7715167498104596],[0.7715167498104596,-0.1543033499620919],[0.7715167498104596,-0.4629100498862757],[0.7715167498104596,1.0801234497346435],[0.7715167498104596,0.7715167498104596],[0.7715167498104596,0.1543033499620919],[0.7715167498104596,0.4629100498862757],[0.1543033499620919,-1.0801234497346435],[0.1543033499620919,-0.7715167498104596],[0.1543033499620919,-0.1543033499620919],[0.1543033499620919,-0.4629100498862757],[0.1543033499620919,1.0801234497346435],[0.1543033499620919,0.7715167498104596],[0.1543033499620919,0.1543033499620919],[0.1543033499620919,0.4629100498862757],[0.4629100498862757,-1.0801234497346435],[0.4629100498862757,-0.7715167498104596],[0.4629100498862757,-0.1543033499620919],[0.4629100498862757,-0.4629100498862757],[0.4629100498862757,1.0801234497346435],[0.4629100498862757,0.7715167498104596],[0.4629100498862757,0.1543033499620919],[0.4629100498862757,0.4629100498862757]])

def get_symbol_map(bits_per_symbol):
    return symbol_maps[bits_per_symbol]

def calc_EbN0(modulator, N0):
    '''
    Calculates EBN0 for given modulator and N0 values.
    Inputs:
    modulator: Modulator object whose constellation is used.
    N0: Float np.array or constant.
    
    Outputs:
    EBN0: EBN0 in decibels. 
    '''
    
    symbols_i = np.array(range(2**modulator.bits_per_symbol), dtype='int')
    constellation = modulator.modulate(symbols_i)
    Es = np.mean(np.abs(constellation)**2)
    EbN0_lin = Es / (modulator.bits_per_symbol * N0)
    EbN0 = 10*np.log10(EbN0_lin)
    return EbN0

def calc_N0(modulator, EBN0):
    '''
    Calculates N0 for given modulator and EBN0 values
    Inputs:
    modulator: Modulator object whose constellation is used.
    EBN0: Float np.array or constant.
    
    Outputs:
    N0: N0 values
    '''
    symbols_i = np.array(range(2**modulator.bits_per_symbol), dtype='int')
    constellation = modulator.modulate(symbols_i)
    Es = np.mean(np.abs(constellation)**2)
    EBN0_lin = 10**(0.1*EBN0)
    N0 = Es/(EBN0_lin*modulator.bits_per_symbol)
    return N0

def main():
    #Testing functions
    import visualize
    
    mod_types = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
    for mod_type in mod_types:
        bits_per_symbol, symbol_map, legend_map = get_modulation_maps(mod_type=mod_type)
        print(mod_type, bits_per_symbol)  
        for i in range(len(symbol_map)):
            print(i, symbol_map[i], legend_map[i])
        print("")
        visualize.visualize_constellation(data=list(symbol_map.values()), labels=list(symbol_map.keys()))

if __name__ == '__main__':
    main()

