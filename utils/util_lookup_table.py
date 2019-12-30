import os
import pickle
from bisect import bisect_left

import numpy as np

UTILS_DIR = os.path.dirname(os.path.realpath(__file__))
def load_lookup_table():
    lookup_table_filename = UTILS_DIR + '/ber_lookup_table.pkl'
    with open(lookup_table_filename, 'rb') as f:
        lookup_table = pickle.load(f)
        return lookup_table


def save_lookup_table(lookup_table):
    lookup_table_filename = UTILS_DIR + '/ber_lookup_table.pkl'
    with open(lookup_table_filename, 'wb') as f:
        pickle.dump(lookup_table, f)


def find_ge(a, key):
    '''Find smallest item greater-than or equal to key.
    Raise ValueError if no such item exists.
    If multiple keys are equal, return the leftmost.

    '''
    i = bisect_left(a, key)
    if i == len(a):
        return None
    else:
        return i


class BER_lookup_table():
    def __init__(self):
        self.err_tolerance = 1e-9  # Indifferent to ber in range (ber, ber+tolerance) so 1e-9 is same as 0

        # Load lookup table from pickle file
        lookup_table_filename = UTILS_DIR + '/ber_lookup_table.pkl'
        with open(lookup_table_filename, 'rb') as f:
            lookup_table = pickle.load(f)
            self.lookup_table = lookup_table
        self.seen_ber = {}
        self.seen_ser = {}
        self.seen_ber_roundtrip = {}
        self.seen_ser_roundtrip = {}
        self.seen_snr_for_ber = {}
        self.seen_snr_for_ser = {}
        self.seen_snr_for_ber_roundtrip = {}
        self.seen_snr_for_ser_roundtrip = {}

        for key in self.lookup_table:
            self.seen_ber[key] = {}
            self.seen_ser[key] = {}
            self.seen_ber_roundtrip[key] = {}
            self.seen_ser_roundtrip[key] = {}
            self.seen_snr_for_ber[key] = {}
            self.seen_snr_for_ser[key] = {}
            self.seen_snr_for_ber_roundtrip[key] = {}
            self.seen_snr_for_ser_roundtrip[key] = {}

    def get_optimal_SNR_for_BER(self, target_ber, bits_per_symbol):
        '''
        Return the smallest SNR in db so that ber in classics using mod type corresponding to bits_per_symbol is less than target_ber + err_tolerance
        '''
        ckey = np.round(np.log10(target_ber + self.err_tolerance), 2)
        if ckey not in self.seen_ber[bits_per_symbol]:
            ck = find_ge(a=-self.lookup_table[bits_per_symbol][:, 1], key=-target_ber - self.err_tolerance)
            if ck is not None:
                curval = self.lookup_table[bits_per_symbol][ck, 0]
                self.seen_ber[bits_per_symbol][ckey] = curval
                return curval
            else:
                raise ValueError('a ber value below target ber was not achieved for any SNR_db in classics!')
        else:
            return self.seen_ber[bits_per_symbol][ckey]

    def get_optimal_SNR_for_SER(self, target_ser, bits_per_symbol):
        '''
        Return the smallest SNR in db so that ser in classics using mod type corresponding to bits_per_symbol is less than target_ber + err_tolerance
        '''
        ckey = np.round(np.log10(target_ser + self.err_tolerance), 2)
        if ckey not in self.seen_ser[bits_per_symbol]:
            ck = find_ge(a=-self.lookup_table[bits_per_symbol][:, 2], key=-target_ser - self.err_tolerance)
            if ck is not None:
                curval = self.lookup_table[bits_per_symbol][ck, 0]
                self.seen_ser[bits_per_symbol][ckey] = curval
                return curval
            else:
                raise ValueError('a ser value below target ser was not achieved for any SNR_db in classics!')
        else:
            return self.seen_ser[bits_per_symbol][ckey]

    def get_optimal_SNR_for_BER_roundtrip(self, target_ber_roundtrip, bits_per_symbol):
        '''
        Return the smallest SNR in db so that ber_roundtrip in classics using mod type corresponding to bits_per_symbol is less than target_ber + err_tolerance
        '''
        ckey = np.round(np.log10(target_ber_roundtrip + self.err_tolerance), 2)
        if ckey not in self.seen_ber_roundtrip[bits_per_symbol]:
            ck = find_ge(a=-self.lookup_table[bits_per_symbol][:, 3], key=-target_ber_roundtrip - self.err_tolerance)
            if ck is not None:
                curval = self.lookup_table[bits_per_symbol][ck, 0]
                self.seen_ber_roundtrip[bits_per_symbol][ckey] = curval
                return curval
            else:
                raise ValueError(
                    'a ber_roundtrip value below target ber_roundtrip was not achieved for any SNR_db in classics!')
        else:
            return self.seen_ber_roundtrip[bits_per_symbol][ckey]

    def get_optimal_SNR_for_SER_roundtrip(self, target_ser_roundtrip, bits_per_symbol):
        '''
        Return the smallest SNR in db so that ser_roundtrip in classics using mod type corresponding to bits_per_symbol is less than target_ber + err_tolerance
        '''
        ckey = np.round(np.log10(target_ser_roundtrip + self.err_tolerance), 2)
        if ckey not in self.seen_ser_roundtrip[bits_per_symbol]:
            ck = find_ge(a=-self.lookup_table[bits_per_symbol][:, 4], key=-target_ser_roundtrip - self.err_tolerance)
            if ck is not None:
                curval = self.lookup_table[bits_per_symbol][ck, 0]
                self.seen_ser_roundtrip[bits_per_symbol][ckey] = curval
                return curval
            else:
                raise ValueError(
                    'a ser_roundtrip value below target ser_roundtrip was not achieved for any SNR_db in classics!')
        else:
            return self.seen_ser_roundtrip[bits_per_symbol][ckey]

    def get_optimal_BER(self, target_SNR, bits_per_symbol):
        '''
        Return ber of classics at this SNR value (in db)
        '''
        ckey = np.round(target_SNR, 1)
        if ckey not in self.seen_snr_for_ber[bits_per_symbol]:
            ck = find_ge(a=self.lookup_table[bits_per_symbol][:, 0], key=target_SNR)
            if ck is not None:
                curval = self.lookup_table[bits_per_symbol][ck, 1]
                self.seen_snr_for_ber[bits_per_symbol][ckey] = curval
                return curval
            else:
                return 0
        else:
            return self.seen_snr_for_ber[bits_per_symbol][ckey]

    def get_optimal_SER(self, target_SNR, bits_per_symbol):
        '''
        Return ber of classics at this SNR value (in db)
        '''
        ckey = np.round(target_SNR, 1)
        if ckey not in self.seen_snr_for_ser[bits_per_symbol]:
            ck = find_ge(a=self.lookup_table[bits_per_symbol][:, 0], key=target_SNR)
            if ck is not None:
                curval = self.lookup_table[bits_per_symbol][ck, 2]
                self.seen_snr_for_ser[bits_per_symbol][ckey] = curval
                return curval
            else:
                return 0
        else:
            return self.seen_snr_for_ser[bits_per_symbol][ckey]

    def get_optimal_BER_roundtrip(self, target_SNR, bits_per_symbol):
        '''
        Return ber of classics at this SNR value (in db)
        '''
        ckey = np.round(target_SNR, 1)
        if ckey not in self.seen_snr_for_ber_roundtrip[bits_per_symbol]:
            ck = find_ge(a=self.lookup_table[bits_per_symbol][:, 0], key=target_SNR)
            if ck is not None:
                curval = self.lookup_table[bits_per_symbol][ck, 3]
                self.seen_snr_for_ber_roundtrip[bits_per_symbol][ckey] = curval
                return curval
            else:
                return 0
        else:
            return self.seen_snr_for_ber_roundtrip[bits_per_symbol][ckey]

    def get_optimal_SER_roundtrip(self, target_SNR, bits_per_symbol):
        '''
        Return ber of classics at this SNR value (in db)
        '''
        ckey = np.round(target_SNR, 1)
        if ckey not in self.seen_snr_for_ser_roundtrip[bits_per_symbol]:
            ck = find_ge(a=self.lookup_table[bits_per_symbol][:, 0], key=target_SNR)
            if ck is not None:
                curval = self.lookup_table[bits_per_symbol][ck, 4]
                self.seen_snr_for_ser_roundtrip[bits_per_symbol][ckey] = curval
                return curval
            else:
                return 0
        else:
            return self.seen_snr_for_ser_roundtrip[bits_per_symbol][ckey]

