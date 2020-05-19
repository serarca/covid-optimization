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
from covid_env import CovidEnvContinuous
import gym
import math

from stable_baselines import A2C
import pprint



# Global variables
simulation_params = {
	'dt':1.0,
	'days': 182.0,
	'policy_freq': 7.0,
}
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))
simulation_params['n_policies'] = int(math.ceil(simulation_params["days"]/simulation_params["policy_freq"]))
baseline = "000120"

region = "Ile-de-France"

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-heuristic", "--heuristic", help="Whether to draw plots")
parser.add_argument("-policy", "--policy", help="Policy")
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
parser.add_argument("-policy_params", "--policy_params", help="Number of M tests")
args = parser.parse_args()


# Read group parameters
with open("./parameters/"+region+".yaml") as file:
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

# Create environment
env = CovidEnvContinuous(universe_params, simulation_params, actions_dict, initialization)


# Construct vector of tests with a heuristic
max_m_tests = [float(args.m_tests) for t in range(simulation_params['n_policies'])]
max_a_tests = [float(args.a_tests) for t in range(simulation_params['n_policies'])]
if args.heuristic == "random":
	a_tests_vec, m_tests_vec = random_partition(env.dynModel, max_a_tests, max_m_tests)
elif args.heuristic == "homogeneous":
	a_tests_vec, m_tests_vec = homogeneous(env.dynModel, max_a_tests, max_m_tests)
elif "age_group" in args.heuristic:
	a_tests_vec, m_tests_vec = all_to_one(env.dynModel, args.heuristic, max_a_tests, max_m_tests)
elif args.heuristic == "no_tests":
	a_tests_vec, m_tests_vec = no_tests(env.dynModel)
elif args.heuristic == "forecasting_heuristic":
    tolerance = 1000000
    max_iterations = 2
    a_tests_vec, m_tests_vec = forecasting_heuristic(env.dynModel, max_a_tests, max_m_tests, h_cap_vec, icu_cap_vec, tolerance, max_iterations)

tests = {
	'a_tests_vec':a_tests_vec,
	'm_tests_vec':m_tests_vec,
}
env.testing(tests)


# Define policy
if args.policy == "constant":
	static_policy = [int(s) for s in args.policy_params]
elif args.policy == "a2c_model":
	model = A2C('MlpPolicy', env, verbose=1)
	model = model.load("./RLModels/a2c_model")
elif args.policy == "baseline":
	static_policy = [int(s) for s in args.policy_params]



# Simulate environment
obs = env.reset()
actions = []
rewards = 0
while True:
	if args.policy in ["constant","baseline"]:
		action = static_policy
	else:
		action = env.multidiscrete_to_action((0,0,0,1,2,0))
	print(action)
	obs, reward, done, info = env.step(action)
	rewards += reward
	actions.append(action)
	if done:
		break

env.dynModel.print_stats()





# Draw plots
dynModel = env.dynModel
time_axis = [i*simulation_params["dt"] for i in range(simulation_params['time_periods']+1)]


groups = dynModel.groups.keys()
plt.figure(1)
for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1)
	plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
	plt.title(group)
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1+len(groups))
	plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
	plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
	plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1+len(groups)*2)
	plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1+len(groups)*3)
	plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
	plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
	plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
	plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1+len(groups)*4)
	plt.plot(time_axis, dynModel.groups[group].H, label="Hospital Bed")
	plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
	plt.plot(time_axis, dynModel.groups[group].D, label="Dead")
	plt.legend(loc='upper right')

plt.subplot(6,2,11)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
plt.legend(loc='upper right')

plt.subplot(6,2,12)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.legend(loc='upper right')



figure = plt.gcf() # get current figure
figure.set_size_inches(6*len(groups),18)
figure.suptitle('Region: %s, Policy: %s, MTests/day: %s, Heuristic: %s'%(region,args.policy,args.m_tests,args.heuristic), fontsize=22)
plt.savefig("results_runs/"+region+"_lp_"+args.policy+"_params_"+args.policy_params+"_m_tests_"+args.m_tests+"_heuristic_"+args.heuristic+".png", dpi = 100)

