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
# from forecasting_heuristic import *
import math
import pprint

from hospital_data_parser.parser import *

def generateOneDynModelPath(daysToLockDown, alpha_contacts):
	n_days = daysToLockDown + daysAfterLockDown

	# Global variables
	simulation_params = {
		'dt':1.0,
		'region': "Ile-de-France",
		'days': n_days,
	}

	# Define time variables
	simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

	# Read group parameters
	with open("./parameters/"+simulation_params["region"]+".yaml") as file:
		# The FullLoader parameter handles the conversion from YAML
		# scalar values to Python the dictionary format
		universe_params = yaml.load(file, Loader=yaml.FullLoader)

	# Read initialization
	with open("./initialization/patient_zero.yaml") as file:
		# The FullLoader parameter handles the conversion from YAML
		# scalar values to Python the dictionary format
		initialization = yaml.load(file, Loader=yaml.FullLoader)

	# Update initialization
	# Put exactly initial_infected infected individuals in age group 40-49. No infected individuals in other groups.
	initialization["age_group_40_49"]["I"] = initialization["age_group_40_49"]["I"] + int(args.initial_infected)
	initialization["age_group_40_49"]["S"] = initialization["age_group_40_49"]["S"] - int(args.initial_infected)
	#initialization["age_group_50_59"]["I"] = initialization["age_group_50_59"]["I"] + int(args.initial_infected)
	#initialization["age_group_50_59"]["S"] = initialization["age_group_50_59"]["S"] - int(args.initial_infected)

	# Read lockdown
	with open("./alphas_action_space/default.yaml") as file:
		# The FullLoader parameter handles the conversion from YAML
		# scalar values to Python the dictionary format
		actions_dict = yaml.load(file, Loader=yaml.FullLoader)

	# Create environment
	mixing_method = {
		"name":"mult",
		"param":0.0,
	}
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
		a_tests_vec, m_tests_vec = forecasting_heuristic(dynModel, max_a_tests, max_m_tests, [dynModel.beds for t in range(len(max_a_tests))], [dynModel.icus for t in range(len(max_a_tests))], tolerance, max_iterations)
	#ICU CAP replaced by single value dynModel.icus
	tests = {
		'a_tests_vec':a_tests_vec,
		'm_tests_vec':m_tests_vec,
	}


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
	
	# Run the model for the whole time range
	for t in range(0,daysToLockDown):# for t in range(0,int(args.lockdown_start)):
		dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], no_lockdown)
	#print("Deaths at time of lockdown:", sum([dynModel.deaths[t] for t in range(0,int(daysToLockDown)+1) if t!=0]))
	
	for t in range(daysToLockDown, daysToLockDown + daysAfterLockDown):# for t in range(int(args.lockdown_start),int(args.n_days)):
		dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], total_lockdown)
	#print("Deaths at the end of time:", sum([dynModel.deaths[t] for t in range(0,int(args.n_days)+1) if t!=0]))
	#print("Deaths after lockdown:", sum([dynModel.deaths[t] for t in range(daysToLockDown,int(args.n_days)+1) if t!=0]))
	
	predicted_deaths_by_group_dict = {}
	deltas_by_group_dict = {}
	groups = dynModel.groups.keys()
	for i,group in enumerate(groups):
		predicted_deaths_by_group_dict[group] = dynModel.groups[group].D[n_days]
	
	predicted_deaths_by_time = []
	for time_per in range(daysToLockDown, daysToLockDown + daysAfterLockDown):
		predicted_deaths_by_time.append(0)
		for i,group in enumerate(groups):
			predicted_deaths_by_time[time_per - daysToLockDown] += dynModel.groups[group].D[time_per] - dynModel.groups[group].D[time_per - 1]
		
	return predicted_deaths_by_time, predicted_deaths_by_group_dict

def processHospData(filePath):
	# Get hospital data and arrange into group specific dataframes
	covidParser = CovidParser(filePath)
	dptCodes = [75, 92, 93, 94, 95, 77, 91, 78]
	covidParser.df_master = filterDataFrame(covidParser.df_master, dptCodes)

	ageBuckets = [9,19,29,39,49,59,69,79,89,90]
	ageGroupStrings = ["age_group_0_9", "age_group_10_19", "age_group_20_29", "age_group_30_39", "age_group_40_49", "age_group_50_59", "age_group_60_69", "age_group_70_79", "age_group_80_plus"]

	# Create dictionary of real hospitalization data by age group. Some timeseries (deaths and recovered) are cumulative
	# TODO: Make all columns non-cumulative
	hosp_data_dfs_by_age_dict = createDFsbyAgeGroup(covidParser.df_master, ageBuckets, ageGroupStrings)
	real_deaths_total = 0
	real_deaths_by_group_dict = {}
	for age_group in ageGroupStrings:
		real_deaths_by_group_dict[age_group] = int(hosp_data_dfs_by_age_dict[age_group].iloc[-1]['dc'])
		real_deaths_total += int(hosp_data_dfs_by_age_dict[age_group].iloc[-1]['dc'])
		
	real_deaths_by_time = []
	for time_per in range(0, daysAfterLockDown):
		real_deaths_by_time.append(0)
		for age_group in ageGroupStrings:
			if (time_per > 0):
				real_deaths_by_time[time_per] += int(hosp_data_dfs_by_age_dict[age_group].iloc[time_per]['dc']) - int(hosp_data_dfs_by_age_dict[age_group].iloc[time_per - 1]['dc'])
			else:
				real_deaths_by_time[time_per] += int(hosp_data_dfs_by_age_dict[age_group].iloc[time_per]['dc'])
	
	return real_deaths_total, real_deaths_by_group_dict, real_deaths_by_time

def report_abs_deltas(deltas_by_group_dict):
	total_abs_val = 0
	for vals in deltas_by_group_dict.values():
		total_abs_val += abs(vals)
	return total_abs_val

def findUpperAndLowerBoundModels(fitting_error_param, daysToLockDown_lb_start, daysToLockDown_ub_start):
	# Find lb
	lb_invalid = 0
	daysToLockDown = daysToLockDown_lb_start - 1
	while lb_invalid == 0:
		daysToLockDown += 1
		predicted_deaths_by_time_lb, predicted_deaths_by_group_dict_lb = generateOneDynModelPath(daysToLockDown, 1)
		for time_per in range(0, daysAfterLockDown):
			if ((predicted_deaths_by_time_lb[time_per] / real_deaths_by_time[time_per]) >= (1 + fitting_error_param)):
				lb_invalid = 1
	daysToLockDown_lb = daysToLockDown - 1
	predicted_deaths_by_time_lb, predicted_deaths_by_group_dict_lb = generateOneDynModelPath(daysToLockDown_lb, 1)
		
	# Find ub
	ub_invalid = 0
	daysToLockDown = daysToLockDown_ub_start + 1
	while ub_invalid == 0:
		daysToLockDown -= 1
		predicted_deaths_by_time_ub, predicted_deaths_by_group_dict_ub = generateOneDynModelPath(daysToLockDown, 1)
		for time_per in range(0, daysAfterLockDown):
			if ((predicted_deaths_by_time_ub[time_per] / real_deaths_by_time[time_per]) <= (1 - fitting_error_param)):
				ub_invalid = 1
	daysToLockDown_ub = daysToLockDown + 1
	predicted_deaths_by_time_ub, predicted_deaths_by_group_dict_ub = generateOneDynModelPath(daysToLockDown_ub, 1)
	
	return daysToLockDown_lb, predicted_deaths_by_time_lb, daysToLockDown_ub, predicted_deaths_by_time_ub

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-heuristic", "--heuristic", help="Chose heuristic")
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
#parser.add_argument("-lockdown_start", "--lockdown_start", help="Day at which lockdown started")
#parser.add_argument("-n_days", "--n_days", help="Total number of days")
parser.add_argument("-initial_infected", "--initial_infected", help="Total number of people initially infected")
args = parser.parse_args()

daysAfterLockDown = 61

real_deaths_total, real_deaths_by_group_dict, real_deaths_by_time = processHospData("hospital_data_parser/donnees-hospitalieres-classe-age-covid19-2020-05-18-19h00.csv")

# Simulate SEIR trajectories for different params
print("------------------")

daysToLockDown_lb, predicted_deaths_by_time_lb, daysToLockDown_ub, predicted_deaths_by_time_ub = findUpperAndLowerBoundModels(0.01, 68, 90)
print("Days to lockdown lb = ", daysToLockDown_lb)
print("Days to lockdown ub = ", daysToLockDown_ub)

time_axis = range(0, daysAfterLockDown)
plt.figure(1)
plt.plot(time_axis, predicted_deaths_by_time_lb, label="LB Prediction")
plt.plot(time_axis, predicted_deaths_by_time_ub, label="UB Prediction")
#plt.plot(time_axis, predicted_deaths_by_time_best_fit, label="Best Fit Prediction")
plt.plot(time_axis, real_deaths_by_time, label="Deaths")
figure = plt.gcf() # get current figure
plt.savefig("results_runs/fitting.pdf")


