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
    def __init__(self,  *, agent_dict, name):
        self.name = name
        self.to_echo = None
        self.actions = None
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
                self.demod = Demodulator(model=model_class, **params) 
            else: # function == 'mod'
                self.mod = Modulator(model=model_class, **params)
 

