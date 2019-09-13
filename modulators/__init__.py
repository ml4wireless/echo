import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from modulator_classic import ModulatorClassic
try:
	from modulator_neural import ModulatorNeural
except Exception as e:
	print("Could not import neural modulator: {}".format(e))
from modulator_cluster import ModulatorCluster
try:
    # Requires torch
    from modulator_polynomial import ModulatorPolynomial
except Exception as e:
    print("Could not import polynomial modulator: {}".format(e))


