import numpy as np
from utils.mod_demod_abstract import Demodulator
from utils.util_data import get_bit_l1_loss

class SingleEntryDemod(Demodulator):
    '''
    This class maintains single entry of a lookup table and supports the following functions:
    update_training_set()
    '''
    def __init__(self, decay_rate, r, _class):
        '''
        inputs:
        decay_rate: Rate at which to decay rewards from previous training sets
        r: Maintain top r rewards along with corresponding actions       
        '''
        self.training_set = None # Numpy array with first column containing inputs, second containing 
                                 # actions and third containing rewards
        self.decay_rate = decay_rate
        self.r = r
        self._class = _class

    def update_training_set(self, inputs, actions, rewards):
        '''
        Inputs:
        inputs: np.array of type complex and shape [N] corresponding to the inputs to demodulator
        actions: np.array of type integer and shape [N] corresponding to actions taken
        rewards: np.array of type float and shape [N] with rewards for the actions. Assume rewards
                 are non-negative
        '''
        if len(inputs) == 0:
            return
        
        if np.min(rewards) < 0:
            raise ValueError('Received minimum reward of ' + str(np.min(rewards)) + 
                             ' while rewards must be non-negative!')
        
        
        new_training_set = np.hstack([inputs[:,None], actions[:,None], rewards[:,None]])
        if self.training_set is not None:
            self.training_set[:,2] *= self.decay_rate
            combined_training_set = np.vstack([self.training_set, new_training_set])
        else:
            combined_training_set = new_training_set
               
        sorted_indices = np.argsort(-combined_training_set[:,2])
        combined_training_set_sorted = combined_training_set[sorted_indices,:]
        self.training_set = combined_training_set_sorted[:self.r,:]


class DemodulatorNeighbors():
    def __init__(self, bits_per_symbol, decay_rate, k, r, explore_prob, 
                 explore_decay_rate, classification_method, score_lambda=1e-4, seed=0, **kwargs):
        '''
        Inputs:
        bits_per_symbol: Number of constellation points is 2**bits_per_symbol
        decay_rate: Rate at which to decay rewards from previous training sets
        k: Number of neighbors used for classification  
        r: Maintain top r (input, action, reward) pairs for each class
        score_lambda: Used to offset divide by zero in score calculation
        explore_prob: Probability of exploring
        explore_decay_rate: Rate by which to decay exploration probability
        method: Classification method to use. Options are vote or softmax
        '''
        self.bits_per_symbol = bits_per_symbol
        self.decay_rate = decay_rate
        self.k = k
        self.r = r
        self.num_classes = 2**bits_per_symbol
        self.single_entries = [SingleEntryDemod(self.decay_rate, self.r, i) for i in range(self.num_classes)]
        self.score_lambda = score_lambda
        self.explore_prob = explore_prob
        self.explore_decay_rate = explore_decay_rate
        if classification_method not in ['vote', 'softmax']:
            raise ValueError("Invalid classification method of {}".format(classification_method))
        self.method = classification_method

        self.demod_class = 'neighbors'
        self.rng = np.random.RandomState(seed)

    def demodulate(self, data_c, mode='explore', **kwargs):
        '''
        Inputs:
        data_c: np.array of type complex and shape [N] corresponding to modulated symbols
        mode: explore or exploit. Explore is used for training 
        '''
        training_data = []
        for s in self.single_entries: 
            if s.training_set is not None: 
                training_data.append(s.training_set)
        if len(training_data) == 0:
            # randomly classify
            return self.rng.randint(low=0, high=self.num_classes, size=len(data_c), dtype='int') 
        
        training_data = np.concatenate(training_data) # Contains training data across all classes
        #training_data = self.rng.permutation(training_data)
        labels_si_g = []
        for _input in data_c:
            if mode == 'explore':
                coin_flip = self.rng.binomial(1, self.explore_prob)
                if coin_flip == 1:
                    prediction = self.rng.randint(low=0, high=self.num_classes)
                    labels_si_g.append(prediction)
                    continue
                    
            prediction = self.prediction_procedure(_input, training_data)
            labels_si_g.append(prediction)
            
        return np.array(labels_si_g, dtype='int')

    def update(self, inputs, actions, data_for_rewards, **kwargs):
        '''
        Inputs:
        inputs: np.array of type complex and shape [N] corresponding to the inputs to demodulator
        actions: np.array of type integer and shape [N] corresponding to actions taken
        rewards: np.array of type float and shape [N] with rewards for the actions. Assume rewards
                 are non-negative
        '''
        if 'mode' not in kwargs:
            raise ValueError("Mode not found")
        if kwargs['mode'].lower() == 'echo':
            preamble = inputs
            inputs = data_for_rewards
            # actions = self.demodulate(data_for_rewards, mode='exploit')
            rewards = 1.0 / (get_bit_l1_loss(labels_si=preamble, labels_si_g = actions,\
                                            bits_per_symbol =self.bits_per_symbol) + 1e-2)

            if np.min(rewards) < 0:
                raise ValueError('Received minimum reward of ' + str(np.min(rewards)) + 
                                 ' while rewards must be non-negative!')

            for _class in range(self.num_classes):
                indices = np.where(actions==_class)
                cur_inputs = inputs[indices]
                cur_actions = actions[indices]
                cur_rewards = rewards[indices]
                self.single_entries[_class].update_training_set(inputs=cur_inputs, 
                                                                actions=cur_actions, 
                                                                rewards=cur_rewards)
            self.explore_prob *= self.explore_decay_rate
        else:
            raise ValueError("Only echo supported.")
        
    def calc_nearest_neighbor_scores(self, _input, training_data):
        '''
        Calculate scores from _input to training_data. Large score indicates more likely to choose.
        A simple knn, not using rewards, might have score function as 1/distance.
        
        Inputs:
        training_data: complex np.array of stored (input, action, reward) data
        _input: Scalar, complex input
        '''
        training_inputs = training_data[:,0]
        # converts to float since array is of type complex and rewards are always non-negative
        training_rewards = np.abs(training_data[:,2]) 
        distances = np.abs(_input - training_inputs)
        scores = training_rewards / (distances + self.score_lambda)
        return scores
    
    def get_classification(self, top_k_neighbors, top_k_scores):
        '''
        Get classification based on neighbors
        
        Inputs:
        top_k_neighbors: complex np.array of (input, action, reward) pairs. Contains top k neighbors
                         for some point
        top_k_scores: np.array of type float. Contains scores for the given neighbors
        '''

        # Majority vote by action
        if self.method == 'vote':
            action_counts = np.bincount(top_k_neighbors[:,1].astype('int'))
            prediction = np.argmax(action_counts)
            
        # Sample action using distribution from softmax on scores. 
        if self.method == 'softmax':
            top_k_scores -= np.mean(top_k_scores) # Normalize because of overflow in exponential
            std = np.std(top_k_scores)
            if std != 0:
                top_k_scores /= std
        
            probabilities = np.exp(top_k_scores) / np.sum(np.exp(top_k_scores))
            prediction = np.random.choice(top_k_neighbors[:,1], p=probabilities)
            
        return int(prediction)
    
    def get_class(self, inputs, actions):
        return actions
    
    
                                    
    def prediction_procedure(self, _input, training_data):
        scores = self.calc_nearest_neighbor_scores(_input, training_data)
        top_k_indices = np.argsort(-scores)[:self.k]
        top_k_neighbors = training_data[top_k_indices]
        top_k_scores = scores[top_k_indices]
        prediction = self.get_classification(top_k_neighbors, top_k_scores)
        return prediction
    
    
    
    def visualize_decision_boundary(self, points_per_dim=10, grid = [-1.5, 1.5], title_string='', show=True):
        '''
        Visualize decision boundary for demodulation by passing grid of 2d plane as input and demodulating 
        in mode 'exploit'
        Returns
        num_constellation_points: Number of unique constellation points in grid
        '''
        #Generate grid
        grid_1d = np.linspace(grid[0], grid[1], points_per_dim) 
        grid_2d = np.squeeze(np.array(list(itertools.product(grid_1d, grid_1d))))
        data_c = grid_2d[:,0] + 1j*grid_2d[:,1]

        labels_si_g= self.demodulate(data_c=data_c, mode = 'exploit')
        unique_labels_si_g = np.unique(labels_si_g)

        #print("Num of symbols in decision boundary: " +  str(unique_labels_si_g.shape[0]))
        for i in range(unique_labels_si_g.shape[0]):
            cur_label = unique_labels_si_g[i]
            cur_data_c = data_c[labels_si_g == cur_label]
            plt.scatter(cur_data_c.real, cur_data_c.imag, s = 10)
            plt.annotate(cur_label, (cur_data_c[cur_data_c.shape[0]//2].real,
                                     cur_data_c[cur_data_c.shape[0]//2].imag))

        plt.title(title_string)      
        plt.show()
    