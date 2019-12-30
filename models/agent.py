# import models.modulator_models
# import models.demodulator_models
# print(models.modulator_models)
from importlib import import_module

from models.demodulator import Demodulator
from models.modulator import Modulator


# def my_import(name):
#     components = name.split('.')
#     mod = __import__(components[0])
#     for comp in components[1:]:
#         mod = getattr(mod, comp)
#     return mod

class Agent():
    def __init__(self, *, name, agent_dict=None, verbose=False):
        self.name = name
        self.to_echo = None
        self.actions = None
        if agent_dict is not None:
            for function in ['mod', 'demod']:
                model = agent_dict['%s_model'%function]
                params = agent_dict.get('%s_params'%function, False)
                assert params, "agent %s %s_params not defined"%(name, function)
                assert params.get('bits_per_symbol', False), "agent %s %s_params['bits_per_symbol'] not defined"%(name, function)
                assert params['bits_per_symbol'] in [1,2,3,4], "agent %s %s_params['bits_per_symbol'] is not 1,2,3, or 4"%(name, function)
                #### LOAD MODEL
                module_name = 'models.%sulator_models.%s'%(function, model.capitalize())
                model_class = getattr(import_module(module_name), model.capitalize())

                if function == 'demod':
                    self.demod = Demodulator(model=model_class, verbose=verbose, **params)
                else:  # function == 'mod'
                    self.mod = Modulator(model=model_class, verbose=verbose, **params)

                weights = agent_dict.get('%s_weights'%function, False)
                if weights and function == 'demod':
                    self.demod.load_weights_file(weights)
                elif weights:  # function == 'mod'
                    self.mod.load_weights_file(weights)

    def set_modulator(self, mod):
        self.mod = mod

    def get_modulator(self):
        return self.mod

    def set_demodulator(self, demod):
        self.demod = demod

    def get_demodulator(self):
        return self.demod

