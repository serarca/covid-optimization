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



from group import SEIR_group, DynamicalModel
# from heuristics import *
from forecasting_heuristic import *
import math
import pprint
import collections


# Global variables
simulation_params = {
	'dt':1.0,
	'days': 182,
	'region': "Ile-de-France",
	'quar_freq': 182,
}
age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49', 
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']


# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))



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
with open("./parameters/"+simulation_params["region"]+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("./initialization/initialization.yaml") as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	initialization = yaml.load(file, Loader=yaml.FullLoader)

# Read lockdown
with open("./alphas_action_space/default.yaml") as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	actions_dict = yaml.load(file, Loader=yaml.FullLoader)


# Move population to infected
for group in initialization:
	change = initialization[group]["S"]*float(args.perc_infected)/100
	initialization[group]["S"] = initialization[group]["S"] - change
	initialization[group]["I"] = initialization[group]["I"] + change


# Load the lockdown policy
with open('./benchmarks/'+args.policy+"_infected_"+args.perc_infected+'.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    policy_file = yaml.load(file, Loader=yaml.FullLoader)
alphas_vec = policy_file['alphas_vec']

# Get mixing method
mixing_method = {
	"name":"mult",
}



############################################
####################    Gradient Descent
############################################
intervention_times = [t*simulation_params['quar_freq'] for t in range(int(simulation_params['days']/simulation_params['quar_freq']))]

x0 = np.zeros(len(intervention_times)*len(age_groups)*2)
def simulate_testing(x):
	m = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
	m_tests_vec = []
	a_tests_vec = []
	for i1,t in enumerate(intervention_times):
		m_tests = {
			age_group: x[i1*len(age_groups) + i2] for i2,age_group in enumerate(age_groups)
		}
		for s in range(int(simulation_params['quar_freq']/simulation_params['dt'])):
			m_tests_vec.append(m_tests)
	for i1,t in enumerate(intervention_times):
		a_tests = {
			age_group: x[i1*len(age_groups) + i2 + len(intervention_times)*len(age_groups)] for i2,age_group in enumerate(age_groups)
		}
		for s in range(int(simulation_params['quar_freq']/simulation_params['dt'])):
			a_tests_vec.append(a_tests)

	for s in range(simulation_params['time_periods']):
		m.take_time_step(m_tests_vec[s], a_tests_vec[s], alphas_vec[s])
	total_reward = m.get_total_reward()
	print("%10.10e"%total_reward)
	print(x)
	return -total_reward

# Add bounds
from scipy.optimize import Bounds,minimize,LinearConstraint
bounds = Bounds(
	np.zeros(len(intervention_times)*len(age_groups)*2),
	np.append(
		np.zeros(len(intervention_times)*len(age_groups)) + float(args.m_tests), 
		np.zeros(len(intervention_times)*len(age_groups)) + float(args.a_tests)
	)
)
# Add linear constraints
A = np.zeros((len(intervention_times)*2,len(intervention_times)*len(age_groups)*2))
for i1 in range(len(intervention_times)*2):
	for i2 in range(len(age_groups)):
		A[i1,i1*len(age_groups)+i2] = 1
lb = np.zeros(len(intervention_times)*2)
ub = np.append(np.zeros(len(intervention_times)) + float(args.m_tests), np.zeros(len(intervention_times)) + float(args.a_tests))
lin_constraint = LinearConstraint(A,lb,ub)


# Optimize
#hess = lambda x: np.zeros((len(intervention_times)*len(age_groups)*2, len(intervention_times)*len(age_groups)*2))
start = time.time()
result = minimize(simulate_testing,x0, method='SLSQP',bounds=bounds,options={'eps':1},constraints=[lin_constraint])
end = time.time()


# Obtain the resulting testing
final_x = result.x
final_m_tests_vec = []
final_a_tests_vec = []
for i1,t in enumerate(intervention_times):
	m_tests = {
		age_group: float(final_x[i1*len(age_groups) + i2]) for i2,age_group in enumerate(age_groups)
	}
	for s in range(int(simulation_params['quar_freq']/simulation_params['dt'])):
		final_m_tests_vec.append(copy.deepcopy(m_tests))
for i1,t in enumerate(intervention_times):
	a_tests = {
		age_group: float(final_x[i1*len(age_groups) + i2 + len(intervention_times)*len(age_groups)]) for i2,age_group in enumerate(age_groups)
	}
	for s in range(int(simulation_params['quar_freq']/simulation_params['dt'])):
		final_a_tests_vec.append(copy.deepcopy(a_tests))


final_value = simulate_testing(final_x)

print("Final Value: ",-final_value)

with open('./gradient_testing/'+args.policy+"_infected_"+args.perc_infected+'.yaml', 'w') as file:
    yaml.dump({
    	"quar_freq":simulation_params['quar_freq'],
    	"m_tests_vec":final_m_tests_vec,
    	"a_tests_vec":final_a_tests_vec,
    	"value":float(-final_value),
    	"days":simulation_params['days'],
    	"time":end-start,
    	"a_tests":float(args.a_tests),
    	"m_tests":float(args.m_tests),
    }, file)


