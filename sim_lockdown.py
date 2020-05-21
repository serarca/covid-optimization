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


from group import SEIR_group, DynamicalModel
from heuristics import *
from forecasting_heuristic import *
import math
import pprint


# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-heuristic", "--heuristic", help="Chose heuristic")
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
parser.add_argument("-lockdown_start", "--lockdown_start", help="Day at which lockdown started")
parser.add_argument("-n_days", "--n_days", help="Total number of days")
parser.add_argument("-initial_infected", "--initial_infected", help="Total number of people initially infected")
args = parser.parse_args()


# Global variables
simulation_params = {
	'dt':1.0,
	'region': "Ile-de-France",
	'days': int(args.n_days),
}


# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))


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

# Update initialization
# Put exactly one infected individual in age group 40-49. No infected individuals in other groups.    
initialization["age_group_40_49"]["I"] = initialization["age_group_40_49"]["I"] + int(args.initial_infected)
initialization["age_group_40_49"]["S"] = initialization["age_group_40_49"]["S"] - int(args.initial_infected)

# Read lockdown
with open("./alphas_action_space/default.yaml") as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	actions_dict = yaml.load(file, Loader=yaml.FullLoader)


# Create environment
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'])

# Define policy
total_lockdown_pattern = "000000000"
total_lockdown = {
	'age_group_%d_%d'%(10*i,10*i+9):actions_dict['age_group_%d_%d'%(10*i,10*i+9)][int(total_lockdown_pattern[i])] for i in range(0,8)
}
total_lockdown['age_group_80_plus'] = actions_dict['age_group_80_plus'][0]
no_lockdown_pattern = "444444422"
no_lockdown = {
	'age_group_%d_%d'%(10*i,10*i+9):actions_dict['age_group_%d_%d'%(10*i,10*i+9)][int(no_lockdown_pattern[i])] for i in range(0,8)
}
no_lockdown['age_group_80_plus'] = actions_dict['age_group_80_plus'][2]

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
	a_tests_vec, m_tests_vec = forecasting_heuristic(dynModel, max_a_tests, max_m_tests, [no_lockdown for t in range(0,int(args.lockdown_start))] + [total_lockdown for t in range(int(args.lockdown_start),int(args.n_days))], [dynModel.beds for t in range(len(max_a_tests))], [dynModel.icus for t in range(len(max_a_tests))], tolerance, max_iterations)
#ICU CAP replaced by single value dynModel.icus
tests = {
	'a_tests_vec':a_tests_vec,
	'm_tests_vec':m_tests_vec,
}




# Run the model for the whole time range
for t in range(0,int(args.lockdown_start)):
	dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], no_lockdown)
print("Deaths at time of lockdown:", sum([dynModel.deaths[t] for t in range(0,int(args.lockdown_start)+1) if t!=0]))
for t in range(int(args.lockdown_start),int(args.n_days)):
	dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], total_lockdown)
print("Deaths at the end of time:", sum([dynModel.deaths[t] for t in range(0,int(args.n_days)+1) if t!=0]))


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
figure.suptitle('Region: %s, MTests/day: %s, Heuristic: %s, Initial Infected: %s, Day of Lockdown: %s, Total Days: %s'%(simulation_params['region'],args.m_tests,args.heuristic, args.initial_infected, args.lockdown_start,args.n_days), fontsize=22)
plt.savefig("results_runs/"+simulation_params['region']+"_m_tests_"+args.m_tests+"_heuristic_"+args.heuristic+"_initialinfected_"+args.initial_infected+"_lockdown_start_"+args.lockdown_start+"_n_days_"+args.n_days+".pdf")


dynModel.get_pandas_summary().to_csv("results_runs/"+simulation_params['region']+"_m_tests_"+args.m_tests+"_heuristic_"+args.heuristic+"_initialinfected_"+args.initial_infected+"_lockdown_start_"+args.lockdown_start+"_n_days_"+args.n_days+".csv")
