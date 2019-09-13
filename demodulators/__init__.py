
import sys, os 
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

from demodulator_classic import DemodulatorClassic
try:
	from demodulator_neural import DemodulatorNeural
except Exception as e:
	print("Could not import neural demodulator: {}".format(e))
from demodulator_cluster import DemodulatorCluster
from demodulator_neighbors import DemodulatorNeighbors
try:
    # Requires torch
    from demodulator_polynomial import DemodulatorPolynomial
except Exception as e:
    print("Could not import polynomial demodulator: {}".format(e))

