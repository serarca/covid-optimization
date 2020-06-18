import yaml
from inspect import getsourcefile
import os.path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
sys.path.insert(0, current_dir+"/heuristics")
import copy
import time

# from heuristics import *
# from forecasting_heuristic import *


import math
import pprint
import collections
import numpy as np

from fast_group import FastDynamicalModel
from aux import *


# Global variables
simulation_params = {
	'dt':1.0,
	'days': 182,
	'region': "Ile-de-France",
}
age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49', 
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]

# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))
bouncing = False


# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
parser.add_argument("-perc_infected", "--perc_infected", help="Percentage of population infected")
parser.add_argument("-policy", "--policy", help="Type of policy")
args = parser.parse_args()

# Change policy
if args.policy == "static":
	simulation_params['quar_freq'] = 182
elif args.policy == "dynamic":
	simulation_params['quar_freq'] = 14


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
	initialization[group]["N"] = initialization[group]["S"] + initialization[group]["E"] + initialization[group]["I"] + initialization[group]["R"]

# Define mixing method
mixing_method = {
    "name":"mult",
    "param_alpha":1.0,
    "param_beta":0.5,
    "param":1.0
}



############################################
####################    Gradient Descent
############################################

max_m_tests = float(args.m_tests)
max_a_tests = float(args.a_tests)
all_activities = ['home','leisure','other','school','transport','work']
rel_activities = ['leisure','other','school','transport','work']
intervention_times = [t*simulation_params['quar_freq'] for t in range(int(simulation_params['days']/simulation_params['quar_freq']))]

# Initialize parameters
x0_testing = np.zeros(len(intervention_times)*len(age_groups)*2) + 0.5
x0_lockdown = np.zeros(len(intervention_times)*len(age_groups)*len(rel_activities)) + 0.5
if bouncing:
	x0_bouncing_icu = np.zeros(len(intervention_times)*len(age_groups)) + 0.5

if bouncing:
	x0 = np.append(np.append(x0_testing, x0_lockdown), x0_bouncing_icu)
else:
	x0 = np.append(x0_testing, x0_lockdown)

# Create dynamical model
fastModel = FastDynamicalModel(universe_params, simulation_params['dt'], mixing_method)
initial_state = state_to_matrix(initialization)

def simulate(x):
	# Extract vector components
	x_m_testing = np.reshape(x[0:len(intervention_times)*len(age_groups)],(len(intervention_times),len(age_groups)))
	x_a_testing = np.reshape(x[len(intervention_times)*len(age_groups):len(intervention_times)*len(age_groups)*2],(len(intervention_times),len(age_groups)))
	x_lockdown = np.reshape(
		x[len(intervention_times)*len(age_groups)*2:
		len(intervention_times)*len(age_groups)*2+len(intervention_times)*len(age_groups)*len(rel_activities)],
		(len(intervention_times),len(age_groups),len(rel_activities))
	)

	# Convert testing variables into real testing
	x_m_testing = x_m_testing/np.sum(x_m_testing)*max_m_tests
	x_a_testing = x_a_testing/np.sum(x_a_testing)*max_a_tests

	if bouncing:
		x_bouncing_icu = np.reshape(
			x[len(intervention_times)*len(age_groups)*2+len(intervention_times)*len(age_groups)*len(rel_activities):
			len(intervention_times)*len(age_groups)*3+len(intervention_times)*len(age_groups)*len(rel_activities)],
			(len(intervention_times),len(age_groups))
		)

		# Convert bouncing variables to percentages
		x_bouncing_perc_icu = x_bouncing_icu/np.sum(x_bouncing_icu)


	state = initial_state
	total_reward = 0
	for t in range(simulation_params['time_periods']):
		if t in intervention_times:
			update_contacts = True
		else:
			update_contacts = False
		
		# Add home activity to lockdown
		x_lockdown_all = np.concatenate((np.array([[1.0]]*len(age_groups)),x_lockdown[int(t/simulation_params['quar_freq']),:,:]), axis=1)
		state, econs = fastModel.take_time_step(
			state, 
			x_m_testing[int(t/simulation_params['quar_freq']),:], 
			x_a_testing[int(t/simulation_params['quar_freq']),:],
			x_lockdown_all,
			update_contacts = update_contacts, 
			B_H = False, 
			B_ICU = False,
			B_H_perc = False,
			B_ICU_perc = x_bouncing_perc_icu[int(t/simulation_params['quar_freq']),:] if bouncing else False,
		)
		total_reward += econs['reward']

	print(total_reward)
	return -total_reward


# Create bounds on the variables
from scipy.optimize import Bounds,minimize,LinearConstraint

if bouncing:
	full_bounds = Bounds(
		np.zeros(len(intervention_times)*len(age_groups)*3+len(intervention_times)*len(age_groups)*len(rel_activities)),
		np.zeros(len(intervention_times)*len(age_groups)*3+len(intervention_times)*len(age_groups)*len(rel_activities)) + 1.0
	)
else:
	full_bounds = Bounds(
		np.zeros(len(intervention_times)*len(age_groups)*2+len(intervention_times)*len(age_groups)*len(rel_activities)),
		np.zeros(len(intervention_times)*len(age_groups)*2+len(intervention_times)*len(age_groups)*len(rel_activities)) + 1.0
	)


import time

t0 = time.time()
result_lockdown = minimize(simulate, x0, method='L-BFGS-B',bounds=full_bounds,options={'eps':10e-8})
t1 = time.time()


# Convert results to vectors 
x_m_testing = np.reshape(result_lockdown.x[0:len(intervention_times)*len(age_groups)],(len(intervention_times),len(age_groups)))
x_a_testing = np.reshape(result_lockdown.x[len(intervention_times)*len(age_groups):len(intervention_times)*len(age_groups)*2],(len(intervention_times),len(age_groups)))
x_lockdown = np.reshape(
	result_lockdown.x[len(intervention_times)*len(age_groups)*2:
	len(intervention_times)*len(age_groups)*2+len(intervention_times)*len(age_groups)*len(rel_activities)],
	(len(intervention_times),len(age_groups),len(rel_activities))
)
# Convert testing variables into real testing
x_m_testing = x_m_testing/np.sum(x_m_testing)*max_m_tests
x_a_testing = x_a_testing/np.sum(x_a_testing)*max_a_tests

if bouncing:
	x_bouncing_icu = np.reshape(
		result_lockdown.x[len(intervention_times)*len(age_groups)*2+len(intervention_times)*len(age_groups)*len(rel_activities):
		len(intervention_times)*len(age_groups)*3+len(intervention_times)*len(age_groups)*len(rel_activities)],
		(len(intervention_times),len(age_groups))
	)

	# Convert bouncing variables to percentages
	x_bouncing_perc_icu = x_bouncing_icu/np.sum(x_bouncing_icu)

# Use these variables to construct dictionaries
alphas = matrix_to_alphas(x_lockdown, simulation_params['quar_freq'])
m_tests = matrix_to_vect_of_dict(x_m_testing, simulation_params['quar_freq'])
a_tests = matrix_to_vect_of_dict(x_a_testing, simulation_params['quar_freq'])
B_ICU_perc = matrix_to_vect_of_dict(x_bouncing_perc_icu, simulation_params['quar_freq']) if bouncing else False

optimal_value = float(-simulate(result_lockdown.x))
print("Optimal Value: ", optimal_value)
print("Time: ", t1-t0)

policy = {
	"alphas":alphas,
	"m_tests":m_tests,
	"a_tests":a_tests,
	"B_H_perc":False,
	"B_ICU_perc":B_ICU_perc,
	"B_H":False,
	"B_ICU":False,
	"initialization":initialization,
	"metadata":{
		"quar_freq":simulation_params['quar_freq'],
		"dt":simulation_params['dt'],
		"days":simulation_params['days'],
		"region":simulation_params['region'],
		"time_periods":simulation_params['time_periods'],
	},
	"optimal_value": optimal_value,
	"mixing_method": mixing_method,
	"max_m_tests": float(args.m_tests),
	"max_a_tests": float(args.a_tests),
	"time": t1-t0,
	"gradient_success": result_lockdown.success,
}


with open('./Results/'+args.policy+"_infected_"+args.perc_infected+"_m_tests_"+args.m_tests+"_a_tests_"+args.a_tests+"_bouncing_"+str(bouncing)+'.yaml', 'w') as file:
    yaml.dump(policy, file)







