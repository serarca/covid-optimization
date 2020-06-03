import yaml
from inspect import getsourcefile
import os.path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = Path(current_dir).parent
print(parent_dir)

sys.path.insert(0, str(parent_dir)+"/heuristics")
sys.path.insert(0, str(parent_dir))


from heuristics import *
from group import SEIR_group, DynamicalModel
from upper_group import DynamicalModelUpper

import math
import pprint


# Global variables
simulation_params = {
	'dt':1.0,
	'days': 30.0,
	'region': "Ile-de-France",
}
age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']


# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))



# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-heuristic", "--heuristic", help="Chose heuristic")
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
parser.add_argument("-policy", "--policy", help="Parameters of policy")
parser.add_argument("-perc_infected", "--perc_infected", help="Percentage of population infected")
args = parser.parse_args()


# Read group parameters
with open("../parameters/"+simulation_params["region"]+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("../initialization/initialization.yaml") as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	initialization = yaml.load(file, Loader=yaml.FullLoader)

# Read lockdown
with open("../alphas_action_space/default.yaml") as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	actions_dict = yaml.load(file, Loader=yaml.FullLoader)


# Move population to infected
for group in initialization:
	change = initialization[group]["S"]*float(args.perc_infected)/100
	initialization[group]["S"] = initialization[group]["S"] - change
	initialization[group]["I"] = initialization[group]["I"] + change

# Define policy
policy = {
	'age_group_%d_%d'%(10*i,10*i+9):actions_dict['age_group_%d_%d'%(10*i,10*i+9)][int(args.policy[i])] for i in range(0,8)
}
policy['age_group_80_plus'] = actions_dict['age_group_80_plus'][int(args.policy[8])]

# Define mixing parameter
mixing_method = {
	"name":'mult'
}


# Create environment
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)


# Construct vector of tests with a heuristic
max_m_tests = [float(args.m_tests) for t in range(simulation_params['time_periods'])]
max_a_tests = [float(args.a_tests) for t in range(simulation_params['time_periods'])]


if args.heuristic == "random":
	groups = []
	for group in dynModel.parameters['seir-groups']:
		population = sum([dynModel.initialization[group][sg] for sg in ["S","E","I","R","Ia","Ips","Ims","Iss","Rq","H","ICU","D"]])
		if population > 0:
			groups.append(group)
	groups.sort()

	a_tests_vec, m_tests_vec = random_partition(dynModel, groups, max_a_tests, max_m_tests)
elif args.heuristic == "homogeneous":
	a_tests_vec, m_tests_vec = homogeneous(dynModel, max_a_tests, max_m_tests)
elif "age_group" in args.heuristic:
	a_tests_vec, m_tests_vec = all_to_one(dynModel, args.heuristic, max_a_tests, max_m_tests)
elif args.heuristic == "no_tests":
	a_tests_vec, m_tests_vec = no_tests(dynModel)
elif args.heuristic == "forecasting_heuristic":
	tolerance = 10
	max_iterations = 10
	death_value = 10000
	a_tests_vec, m_tests_vec = forecasting_heuristic(dynModel, max_a_tests, max_m_tests, alphas_vec, [dynModel.beds for t in range(len(max_a_tests))], [dynModel.icus for t in range(len(max_a_tests))], tolerance, max_iterations, death_value)
#ICU CAP replaced by single value dynModel.icus
tests = {
	'a_tests_vec':a_tests_vec,
	'm_tests_vec':m_tests_vec,
}


# Run the model for the whole time range
for t in range(simulation_params['time_periods']):
	dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], policy)

# Print model stats
dynModel.print_stats()

# Now the upper
dynModelUpper = DynamicalModelUpper(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
# Run the model for the whole time range
for t in range(simulation_params['time_periods']):
	dynModelUpper.take_time_step(m_tests_vec[t], a_tests_vec[t], policy)
# Print model stats
dynModelUpper.print_stats()

