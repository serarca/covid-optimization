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
from upper_bound_dynamics import DynamicalModelUpper
import math
import pprint
import yaml
from matplotlib.ticker import PercentFormatter




# Global variables
simulation_params = {
	'dt':1.0,
	'days': 182.0,
	'region': "Ile-de-France",
}
age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))



# Parse parameters
parser = argparse.ArgumentParser()
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
	initialization[group]["S"] = initialization[group]["S"] - 2* change
	initialization[group]["I"] = initialization[group]["I"] + change
	initialization[group]["E"] = initialization[group]["E"] + change


# Define policy
policy = {
	'age_group_%d_%d'%(10*i,10*i+9):actions_dict['age_group_%d_%d'%(10*i,10*i+9)][int(args.policy[i])] for i in range(0,8)
}
policy['age_group_80_plus'] = actions_dict['age_group_80_plus'][int(args.policy[8])]
alphas_vec = [policy for t in range(simulation_params['time_periods'])]

# Define mixing parameter
mixing_method = {
	"name":'mult'
}

# Load intervals
with open('intervals_policy_%s_perc_infected_%s_a_tests_%s_m_tests_%s.yaml'%(args.policy, args.perc_infected, args.a_tests, args.m_tests)) as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	intervals = yaml.load(file, Loader=yaml.FullLoader)




# Create environment
dynModel = DynamicalModelUpper(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, alphas_vec, intervals)


model = dynModel.construct_model(90, float(args.m_tests),float(args.a_tests))


# Construct optimal policy
test_policy = {
	'm_tests':[],
	'a_tests':[]
}
for t in range(90):
	p_m = {}
	p_a = {}
	for group in dynModel.groups:
		p_m[group] = float(dynModel.groups[group].m_tests[t].x)
		p_a[group] = float(dynModel.groups[group].a_tests[t].x)
	test_policy['m_tests'].append(p_m)
	test_policy['a_tests'].append(p_a)

with open('tests_policy_%s_perc_infected_%s_a_tests_%s_m_tests_%s.yaml'%(args.policy, args.perc_infected, args.a_tests, args.m_tests), 'w') as file:
    yaml.dump(test_policy, file)



