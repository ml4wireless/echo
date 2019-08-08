import numpy as np 

SEED = 9
rs = np.random.RandomState(SEED)

#######################
# SAMPLERS
####################### 
# If the parameter is specified as a distribution with a sample method, the parameter is sampled
# [X] checked that returns different random integer for >= 100 times from start
class ListSampler():
    def __init__(self, l):
        self.generator = np.random.RandomState(rs.randint(1000000000)) 
        self.l = l
    def sample(self):
        return self.l[self.generator.randint(len(self.l))]

class UniformSampler():
    def __init__(self, low, high=None):
        self.generator = np.random.RandomState(rs.randint(1000000000)) 
        if high is None:
            self.high=low
        else:
            self.high = high
        self.low = low
    def sample(self):
        s = self.generator.uniform(low=self.low, high=self.high)
        # print(s)
        return s

class IntUniformSampler():
    def __init__(self, low, high=None):
        self.generator = np.random.RandomState(rs.randint(1000000000)) 
        self.high = high
        self.low = low
    def sample(self):
        if self.high is None:
            return self.low
        s = self.generator.randint(low=self.low, high=self.high)
        # print(s)
        return s

class LogUniformSampler():
    def __init__(self, low, high=None):
        self.generator = np.random.RandomState(rs.randint(1000000000)) 
        if high is None:
            self.high=low
        else:
            self.high = high
        self.low = low
    def sample(self):
        s = np.exp(self.generator.uniform(low=np.log(self.low), high=np.log(self.high)))
        # print(s)
        return s

class GenericUniformSampler():
    ''' Default is the same as UniformSampler. '''
    def __init__(self, low, high=None, pre_func=lambda x:x, post_func=lambda x:x):
        self.generator = np.random.RandomState(rs.randint(1000000000)) 
        self.pre_func = pre_func
        self.post_func = post_func
        if high is None:
            self.high=low
        else:
            self.high = high
        self.high = self.pre_func(self.high)
        self.low = self.pre_func(low)
    def sample(self):
        s = self.generator.uniform(low=self.low, high=self.high)
        # print(s)
        return self.post_func(s)
