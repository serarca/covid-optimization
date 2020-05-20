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
sys.path.insert(0, "./gym-covid/gym_covid/envs")

from group import SEIR_group, DynamicalModel
from heuristics import *
<<<<<<< HEAD
from forecasting_heuristic import *
from covid_env import CovidEnvContinuous
import gym
=======
>>>>>>> 7716224690d66ab6362bcd08882d5599281c8a5f
import math
import pprint


# Global variables
simulation_params = {
	'dt':1.0,
	'days': 182.0,
	'region': "Ile-de-France",
}


# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))



# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-heuristic", "--heuristic", help="Chose heuristic")
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
parser.add_argument("-policy_params", "--policy_params", help="Number of M tests")
args = parser.parse_args()


# Read group parameters
<<<<<<< HEAD
with open("./parameters/"+region+".yaml") as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	universe_params = yaml.load(file, Loader=yaml.FullLoader)
=======
with open("./parameters/"+simulation_params["region"]+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)
>>>>>>> 7716224690d66ab6362bcd08882d5599281c8a5f

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


# Create environment
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'])


# Construct vector of tests with a heuristic
max_m_tests = [float(args.m_tests) for t in range(simulation_params['time_periods'])]
max_a_tests = [float(args.a_tests) for t in range(simulation_params['time_periods'])]
if args.heuristic == "random":
<<<<<<< HEAD
	groups = []
	for group in env.dynModel.parameters['seir-groups']:
		population = sum([env.dynModel.initialization[group][sg] for sg in ["S","E","I","R","Ia","Ips","Ims","Iss","Rq","H","ICU","D"]])
		if population > 0:
			groups.append(group)
	groups.sort()

	a_tests_vec, m_tests_vec = random_partition(env.dynModel, groups, max_a_tests, max_m_tests)
=======
	a_tests_vec, m_tests_vec = random_partition(dynModel, max_a_tests, max_m_tests)
>>>>>>> 7716224690d66ab6362bcd08882d5599281c8a5f
elif args.heuristic == "homogeneous":
	a_tests_vec, m_tests_vec = homogeneous(dynModel, max_a_tests, max_m_tests)
elif "age_group" in args.heuristic:
	a_tests_vec, m_tests_vec = all_to_one(dynModel, args.heuristic, max_a_tests, max_m_tests)
elif args.heuristic == "no_tests":
	a_tests_vec, m_tests_vec = no_tests(dynModel)
elif args.heuristic == "forecasting_heuristic":
<<<<<<< HEAD
	tolerance = 10
	max_iterations = 10
	a_tests_vec, m_tests_vec = forecasting_heuristic(env.dynModel, max_a_tests, max_m_tests, [env.dynModel.beds for t in range(len(max_a_tests))], [env.dynModel.icus for t in range(len(max_a_tests))], tolerance, max_iterations)
#ICU CAP replaced by single value dynModel.icus
=======
    tolerance = 10
    max_iterations = 10
    a_tests_vec, m_tests_vec = forecasting_heuristic(dynModel, max_a_tests, max_m_tests, h_cap_vec, icu_cap_vec, tolerance, max_iterations)
# Put tests in dictionary
>>>>>>> 7716224690d66ab6362bcd08882d5599281c8a5f
tests = {
	'a_tests_vec':a_tests_vec,
	'm_tests_vec':m_tests_vec,
}


# Define policy
static_alpha = {
	'age_group_%d'%i:actions_dict['age_group_%d'%i][int(args.policy_params[i-1])] for i in range(1,7)
}

# Run the model for the whole time range
for t in range(simulation_params['time_periods']):
	dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], static_alpha)

# Print model stats
dynModel.print_stats()


# Draw plots
time_axis = [i*simulation_params["dt"] for i in range(simulation_params['time_periods']+1)]

groups = dynModel.groups.keys()
plt.figure(1)
for i,group in enumerate(groups):
	plt.subplot(7,len(groups),i+1)
	plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
	plt.title(group)
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(7,len(groups),i+1+len(groups))
	plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
	plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
	plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(7,len(groups),i+1+len(groups)*2)
	plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(7,len(groups),i+1+len(groups)*3)
	plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
	plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
	plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
	plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(7,len(groups),i+1+len(groups)*4)
	plt.plot(time_axis, dynModel.groups[group].H, label="Hospital Bed")
	plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
	plt.plot(time_axis, dynModel.groups[group].D, label="Dead")
	plt.legend(loc='upper right')


for i,group in enumerate(groups):
	plt.subplot(7,len(groups),i+1+len(groups)*5)
	plt.plot(range(0,int(simulation_params['time_periods'])), re_change_order(m_tests_vec)[group], label="M Tests")
	plt.plot(range(0,int(simulation_params['time_periods'])), re_change_order(a_tests_vec)[group], label="A Tests")
	plt.legend(loc='upper right')

plt.subplot(7,2,13)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
plt.legend(loc='upper right')

plt.subplot(7,2,14)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.legend(loc='upper right')


figure = plt.gcf() # get current figure
figure.set_size_inches(7*len(groups),18)
figure.suptitle('Region: %s, Policy: %s, MTests/day: %s, Heuristic: %s'%(simulation_params['region'],args.policy_params,args.m_tests,args.heuristic), fontsize=22)
plt.savefig("results_runs/"+simulation_params['region']+"_params_"+args.policy_params +"_m_tests_"+args.m_tests+"_heuristic_"+args.heuristic+".pdf")
