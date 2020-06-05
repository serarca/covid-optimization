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
from interval_dynamics import DynamicalModelInterval
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

print(initialization)

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



# Create environment
dynModel = DynamicalModelInterval(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, alphas_vec)



iters = 90
warm_start = False
for t in range(iters):
	warm_start = dynModel.construct_model(t, float(args.m_tests), float(args.a_tests), warm_start)


intervals = {
	"S_L":{},
	"S_U":{},
	"IR_L":{},
	"IR_U":{},
}
for group in age_groups:
	intervals["S_L"][group] = dynModel.groups[group].S_L
	intervals["S_U"][group] = dynModel.groups[group].S_U
	intervals["IR_L"][group] = dynModel.groups[group].IR_L
	intervals["IR_U"][group] = dynModel.groups[group].IR_U

with open('intervals_policy_%s_perc_infected_%s_a_tests_%s_m_tests_%s.yaml'%(args.policy, args.perc_infected, args.a_tests, args.m_tests), 'w') as file:
    yaml.dump(intervals, file)


# Now we draw plots to check how big are the gaps
plt.figure(1)
ax = plt.subplot(1,2,1)
time_axis = list(range(iters))
for i,name in enumerate(age_groups):
	plt.plot(time_axis, (np.array(intervals["S_U"][name])-np.array(intervals["S_L"][name]))/np.array(intervals["S_U"][name]), label=name)
	plt.legend(loc='upper right')
plt.title("S Gap")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))


ax = plt.subplot(1,2,2)
time_axis = list(range(iters))
for i,name in enumerate(age_groups):
	plt.plot(time_axis, (np.array(intervals["IR_U"][name])-np.array(intervals["IR_L"][name]))/np.array(intervals["IR_U"][name]), label=name)
	plt.legend(loc='upper right')
plt.title("IR Gap")
ax.yaxis.set_major_formatter(PercentFormatter(1.0))


figure = plt.gcf() 
figure.set_size_inches(7*2,7)

name_file = 'intervals_policy_%s_perc_infected_%s_a_tests_%s_m_tests_%s.pdf'%(args.policy, args.perc_infected, args.a_tests, args.m_tests)

plt.savefig(name_file)



# # Check how the bounds behave
# name = "age_group_50_59"
# group = dynModel.groups[group]
# print(group.S_U)
# ind_S_U = []
# for i in range(len(group.S_U)):
# 	calc = group.S_U[0] - group.parameters['beta']*sum([group.IR_L[t]*group.S_L[t] for t in range(i)])
# 	ind_S_U.append(calc)

# print(ind_S_U)



# S_U = {}
# S_L = {}
# for age_group in age_groups:
# 	S_U[age_group] = np.array(dynModel.groups[age_group].S_U)
# 	S_L[age_group] = np.array(dynModel.groups[age_group].S_L)


# # # Construct a series where all testing is given to one group
# S = defaultdict(dict)
# for age_group in age_groups:
# 	m_tests = {group:0 for group in dynModel.groups}
# 	n_tests = {group:0 for group in dynModel.groups}
# 	m_tests[age_group] = float(args.m_tests)
# 	dynModelDet = DynamicalModelInterval(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, alphas_vec)
# 	for t in range(iters-1):
# 		dynModelDet.take_time_step(m_tests, n_tests, policy)
# 	for age_group2 in age_groups:
# 		S[age_group2][age_group] = np.array(dynModelDet.groups[age_group2].S)
# m_tests = {group:0 for group in dynModelDet.groups}
# n_tests = {group:0 for group in dynModelDet.groups}
# dynModelDet = DynamicalModelInterval(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, alphas_vec)
# for t in range(iters-1):
# 	dynModelDet.take_time_step(m_tests, n_tests, policy)
# for age_group2 in age_groups:
# 	S[age_group2]["NT"] = np.array(dynModelDet.groups[age_group2].S)

# print(S)

# # Make comparisons
# for age_group in age_groups:
# 	for age_group2 in age_groups:
# 		print(np.min((S_U[age_group]-S[age_group][age_group2])/S_U[age_group]))
# 		print(np.min((S[age_group][age_group2]-S_L[age_group])/S[age_group][age_group2]))

# print("Width")
# for age_group in age_groups:
# 	print(np.max((S_U[age_group]-S_L[age_group])/S_U[age_group]))




# # # Draw plots
# # time_axis = [i*simulation_params["dt"] for i in range(iters)]

# # plt.figure(1)

# # for i,name in enumerate(age_groups):
# # 	plt.subplot(1,len(age_groups),i+1)
# # 	plt.plot(time_axis, S_U[name]-S_L[name], label="S_U-S_L")
# # 	plt.plot(time_axis, S[name] - S_L[name], label="S_%s-S_L"%name)
# # 	plt.legend(loc='upper right')

# # figure = plt.gcf() 
# # figure.set_size_inches(7*len(age_groups),7)

# # plt.savefig("result.pdf")

# print(S["age_group_50_59"])
# print(S_U["age_group_50_59"])


# # print(warm_start["S_L"])





# # dDebug
# # Construct set of variables that should work
# age_group = "age_group_50_59"
# m_tests = {group:0 for group in dynModel.groups}
# n_tests = {group:0 for group in dynModel.groups}
# m_tests[age_group] = float(args.m_tests)
# print(m_tests)
# dynModelDet = DynamicalModelInterval(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, alphas_vec)
# for t in range(iters+1):
# 	dynModelDet.take_time_step(m_tests, n_tests, policy)

# start = {
# 	"z":defaultdict(dict),
# 	"m":defaultdict(dict),
# 	"S":defaultdict(dict),
# 	"I":defaultdict(dict),
# 	"IR":defaultdict(dict),
# }
# for group in age_groups:
# 	for t in range(3):
# 		start["m"][group][t] = m_tests[group]
# 		start["z"][group][t] = dynModelDet.groups[group].total_contacts[t]
# 	t = 3
# 	start["S"][group][t] = dynModelDet.groups[group].S[t]
# 	start["I"][group][t] = dynModelDet.groups[group].I[t]
# 	start["IR"][group][t] = dynModelDet.groups[group].IR[t]

# dynModel.construct_model(3, float(args.m_tests), float(args.a_tests), warm_start=False, force = start)


# S_U = {}
# S_L = {}
# for age_group in age_groups:
# 	S_U[age_group] = np.array(dynModel.groups[age_group].S_U)
# 	S_L[age_group] = np.array(dynModel.groups[age_group].S_L)
# print(S["age_group_50_59"])
# print(S_U["age_group_50_59"])





