###LOSS PASSING###
from utils.util_data import integers_to_symbols
from utils.util_data import add_complex_awgn as add_awgn
from typing import List
import numpy as np

def get_random_preamble(n, bits_per_symbol):
    integers = np.random.randint(low=0,high=2**bits_per_symbol, size=[n])
    return integers_to_symbols(integers, bits_per_symbol)

###LOSS PASSING###
def train(*,
        agents,
        bits_per_symbol:int,
        batch_size:int,
        num_iterations:int,
        results_every:int,
        SNR_db:float,
        signal_power=float,
        plot_callback,
        evaluate_callback,
        **kwargs   
    ):
    if kwargs['verbose']:
        print("loss_passing train.py")
    
    A = agents[0]

    batches_sent = 0
    for i in range(num_iterations+1):
        
        preamble = get_random_preamble(batch_size, bits_per_symbol)
        ##MODULATE/action
        c_signal_forward = A.mod.modulate(preamble, mode='explore', dtype='complex')
        actions=c_signal_forward
        ##CHANNEL
        c_signal_forward_noisy = add_awgn(c_signal_forward, SNR_db = SNR_db, signal_power=signal_power)
        ##DEMODULATE/update and pass loss to mod
        A.demod.update(c_signal_forward_noisy, preamble)
        preamble_halftrip = A.demod.demodulate(c_signal_forward_noisy)
        A.mod.update(preamble, actions, preamble_halftrip)
        batches_sent += 1 
    

        ############### STATS ##########################
        if i%results_every == 0 or i == num_iterations:
            new_kwargs = {**kwargs, 
                'protocol':"loss_passing",
                'agents':agents,
                'bits_per_symbol':bits_per_symbol, 
                'SNR_db':SNR_db,  
                'signal_power':signal_power, 
                'num_iterations':num_iterations,
                'results_every':results_every,'batch_size':batch_size,
                'batches_sent':batches_sent, 'iteration':i,
            }
            evaluate_callback(**new_kwargs)
            plot_callback(**new_kwargs)

            
