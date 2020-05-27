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
	'quar_freq': 182
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




# Create environment
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'])


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
	a_tests_vec, m_tests_vec = forecasting_heuristic(dynModel, max_a_tests, max_m_tests, [static_alpha for t in range(simulation_params['time_periods'])], [dynModel.beds for t in range(len(max_a_tests))], [dynModel.icus for t in range(len(max_a_tests))], tolerance, max_iterations)
#ICU CAP replaced by single value dynModel.icus
tests = {
	'a_tests_vec':a_tests_vec,
	'm_tests_vec':m_tests_vec,
}




############################################
####################    Gradient Descent
############################################
all_activities = ['home','leisure','other','school','transport','work']
rel_activities = ['leisure','other','school','transport','work']
intervention_times = [t*simulation_params['quar_freq'] for t in range(int(simulation_params['days']/simulation_params['quar_freq']))]
x0 = np.zeros(len(intervention_times)*len(age_groups)*len(rel_activities)) + 0.5


# Write a function that does the same with testing and only testing
def simulate(x):
	m = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'])
	alphas_vec = []
	for i1,t in enumerate(intervention_times):
		alphas = {
			age_group:{
				activity: x[i1*len(age_groups)*len(rel_activities) + i2*len(rel_activities) + i3] for i3,activity in enumerate(rel_activities)
			} for i2,age_group in enumerate(age_groups)
		}
		for j,age_group in enumerate(age_groups):
			alphas[age_group]['home'] = 1.0
		for s in range(int(simulation_params['quar_freq']/simulation_params['dt'])):
			alphas_vec.append(alphas)
	# Run the model for the whole time range
	for s in range(simulation_params['time_periods']):
		m.take_time_step(m_tests_vec[s], a_tests_vec[s], alphas_vec[s])
	total_reward = m.get_total_reward()
	print("%10.10e"%total_reward)
	print(x)
	return -total_reward


from scipy.optimize import Bounds,minimize
bounds = Bounds(np.zeros(len(intervention_times)*len(age_groups)*len(rel_activities)),np.zeros(len(intervention_times)*len(age_groups)*len(rel_activities)) + 1.0)

# Optimize
start = time.time()
result = minimize(simulate,x0, method='L-BFGS-B',bounds=bounds,options={'eps':10e-6})
end = time.time()

final_x = result.x
final_alphas_vec = []
for i1,t in enumerate(intervention_times):
	alphas = {
		age_group:{
			activity: float(final_x[i1*len(age_groups)*len(rel_activities) + i2*len(rel_activities) + i3]) for i3,activity in enumerate(rel_activities)
		} for i2,age_group in enumerate(age_groups)
	}
	for j,age_group in enumerate(age_groups):
		alphas[age_group]['home'] = 1.0
	for s in range(int(simulation_params['quar_freq']/simulation_params['dt'])):
		final_alphas_vec.append(copy.deepcopy(alphas))

print("Final Value: ",simulate(final_x))
print("Actions Vector: ", final_x)

with open('./benchmarks/'+args.policy+"_infected_"+args.perc_infected+'.yaml', 'w') as file:
    yaml.dump({
    	"alphas_vec":final_alphas_vec,
    	"quar_freq":simulation_params['quar_freq'],
    	"value":-float(simulate(final_x)),
    	"days":simulation_params['days'],
    	"time":end-start,
    }, file)


