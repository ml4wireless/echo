from abc import ABCMeta, abstractmethod
from utils.visualize import visualize_constellation
from utils.visualize import visualize_decision_boundary

# This is an abstract class for a modulator. Concrete modulator classes inherit from this and should implement its methods.
class Modulator:

    #### Include a mod_class attribute. E.g, ModulatorNeural().mod_class is equal to 'neural' ###
    __metaclass__ = ABCMeta
    @abstractmethod
    def update(self, preamble_si, actions, labels_si_g, **kwargs): #TODO specify provided kwargs
        pass

    @abstractmethod
    def modulate(self, data_si, **kwargs):
        pass

    def visualize(self, preamble_si, title_string=None):
        if not title_string:
            title_string = "Modulator %s"%self.demod_class.capitalize()
        data_si = np.arange(2**self.bits_per_symbol)
        data_m = self.modulate(data_si=preamble_si, mode='explore')
        data_m_centers = self.modulate(data_si=preamble_si, mode='exploit')
        args = {"data":data_m,
                "data_centers":data_m_centers,
                "labels":preamble_si,
                "legend_map":{i:i for i in range(2**self.bits_per_symbol)},
                "title_string":title_string,
                "show":True}
        visualize_constellation(**args)




#This is an abstract class for a demodulator. Concrete demodulator classes inherit from this and should implement its methods.
class Demodulator:

    #### Include a demod_class attribute. E.g, ModulatorNeural().demod_class is equal to 'neural' ###
    __metaclass__ = ABCMeta
    @abstractmethod
    def update(self, inputs, actions, data_for_rewards, **kwargs): #TODO
        pass

    @abstractmethod
    def demodulate(self, data_c, **kwargs):
        pass

    def visualize(self, title_string=None):
        if not title_string:
            title_string = "Demodulator %s"%self.demod_class.capitalize()
        visualize_decision_boundary(self, points_per_dim=100, title_string=title_string)()
