#K-means class
import numpy as np
from scipy import stats

class Kmeans():
    '''Implements lloyd algorithm for k-means clustering'''

    def __init__(self,k,ref_means=None):
        '''
        Initializer for class
        Inputs:
        ref_means: the means of clusters indexed by 0,...,k-1 that we want to try and match
        k: Number of clusters
        
        '''
        self.k = k
        self.means = None 
        self.initialized = False
        self.ref_means = ref_means
    
    
    def initialize(self, data, hard=False):
        '''
        Implements the k++ initialization algorithm 
            Inputs
                data: complex valued np.array of shape [N]
                hard: enforces maximum instead of pmf sampling
            Outputs 
                complex valued np.array of shape [k]   
        '''
        N = data.shape[0] # number of data points
        range_N = np.arange(N)
        means = [data[np.random.randint(N)]]
        dists = float('inf')*np.ones(N) # holds distance of all points to closest mean

        for k in range(self.k-1):            
            # refresh list of minimal distances
            mean = means[-1] 
            dists = np.minimum(dists, abs(data - mean))

            if (hard): # choose next mean according to max distance 
                means.append(data[np.argmax(dists)])
    
            else: # samples next mean according to pmf ~ distance
                probs = dists/float(np.sum(dists))
                pmf = stats.rv_discrete(values=(range_N, probs))
                means.append(data[pmf.rvs(1)])
                 
        means = np.array(means)
        self.means = means
        self.initialized = True 
        return means

    def iterate(self, data, num_iterations):
        ''' 
        Executes Lloyd's algorithm on the provided data with the initialized means for i iterations
        Inputs:
            data: complex valued np.array of shape [N]
            num_iterations: number of iterations
        Output:
            assign: np.array of type integer and shape [N] in which each element is the cluster (index of the mean)\
            assigned to the respective symbol
        '''
        if (not self.initialized):
            raise Exception("Not initialized. Run initialize() first") 
        N = data.shape[0] 
        means = self.means      
        for i in range(num_iterations):
            # assign points to mean
                assign = np.argmin(abs(data[:, None] - means[None, :]), axis=1)               
                means = np.array([np.mean(data[assign==k]) for k in range(self.k)]) 
        self.means = means        
        return assign
    
    