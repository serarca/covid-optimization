import yaml
import argparse
from heuristics import *
# Parse data
parser = argparse.ArgumentParser()
parser.add_argument("-data", "--data", help="Source file for data")
parser.add_argument("-heuristic", "--heuristic", help="Name of heuristic")
args = parser.parse_args()

# Import group module
from inspect import getsourcefile
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
from group import SEIR_group, DynamicalModel


with open("../parameters/"+args.data) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)


# Set up parameters of simulation
dt = 0.1
total_time = 100
# Choose 

# Total time periods
time_periods = int(round(total_time/dt))

# Capacity of beds
h_cap_vec = [parameters['global-parameters']['C_H'] for t in range(time_periods)]
icu_cap_vec = [parameters['global-parameters']['C_ICU'] for t in range(time_periods)]

# Number of tests
max_a_tests = [parameters['global-parameters']['A_tests'] for t in range(time_periods)]
max_m_tests = [parameters['global-parameters']['M_tests'] for t in range(time_periods)]

# Choose heuristic
exec("%s = %s" % ("heuristic",args.heuristic))



# Create model
dynModel = DynamicalModel(parameters, dt, time_periods)

# Get heuristic
a_tests_vec, m_tests_vec = heuristic(dynModel, max_a_tests, max_m_tests)


dynModel.simulate(m_tests_vec, a_tests_vec, h_cap_vec, icu_cap_vec)

print("Objective value is: %f" % dynModel.get_objective_value())