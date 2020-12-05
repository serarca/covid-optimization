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
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir+"/heuristics")
sys.path.insert(0, parent_dir+"/fast_gradient")
sys.path.insert(0, current_dir+"/heuristics/LP-Models")
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from group import SEIR_group, DynamicalModel
import math
import pprint
import pandas as pd
import pickle
import numpy as np
from fast_group import FastDynamicalModel
from aux import *
from scipy.optimize import Bounds,minimize,LinearConstraint

parser = argparse.ArgumentParser(description='Arguments')
parser.add_argument('--delta', action="store", dest='delta', type=float)
parser.add_argument('--icus', action="store", dest='icus', type=int)
parser.add_argument('--eta', action="store", dest='eta', type=float)
parser.add_argument('--groups', action="store", dest='groups', type=str)
parser.add_argument('--xi', action="store", dest='xi', type=float)
parser.add_argument('--a_tests', action="store", dest='a_tests', type=int)
parser.add_argument('--m_tests', action="store", dest='m_tests', type=int)

args = parser.parse_args()

run_params = {
	"groups":args.groups,
	"params_to_try":{
		"delta_schooling":[args.delta],
		"icus":[args.icus],
		"eta":[args.eta],
		"tests":[[args.m_tests, args.a_tests]],
		"xi":[args.xi],
		"testing":["homogeneous"]
	}
}

print(run_params)
quar_freq = 1


thresholds_to_try = {
	"icu_utilization_rate":[0.1*(i+1) for i in range(11)],
	"sevenday_avg_newICU":[10*(i+1) for i in range(41)] 
}
lockdowns_to_try = {
	"low_activity":[0.1*i for i in range(11)],
	"high_activity":[0.1*i for i in range(11)]
}


params_to_try = run_params["params_to_try"]
groups = run_params["groups"]



cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]

# Global variables
simulation_params = {
	'dt':1.0,
	'days': 90.0,
	'region': "fitted",
	'heuristic': 'benchmark',
	'mixing_method': {'name': 'multi'}
}
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

if groups == "all":
	age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']

	# Read group parameters
	with open("../parameters/fitted.yaml") as file:
	    universe_params = yaml.load(file, Loader=yaml.FullLoader)

	# Read initialization
	with open("../initialization/oct21.yaml") as file:
		initialization = yaml.load(file, Loader=yaml.FullLoader)
		start_day = 0

	# Read econ parameters
	with open("../parameters/econ.yaml") as file:
		econ_params = yaml.load(file, Loader=yaml.FullLoader)

# elif groups == "one":
# 	age_groups = ["all_age_groups"]

# 	# Read group parameters
# 	with open("../parameters/one_group_fitted.yaml") as file:
# 	    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# 	# Read initialization
# 	with open("../initialization/60days_one_group.yaml") as file:
# 		initialization = yaml.load(file, Loader=yaml.FullLoader)
# 		start_day = 60

# 	# Read econ parameters
# 	with open("../parameters/one_group_econ.yaml") as file:
# 		econ_params = yaml.load(file, Loader=yaml.FullLoader)

# Read lower bounds
with open("../lower_bounds/fitted.yaml") as file:
	lower_bounds = yaml.load(file, Loader=yaml.FullLoader)




cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]
activities = ['home','leisure','other','school','transport','work']
rel_activities = ['leisure','other','school','transport','work']


# age_groups = ['age_group_30_39']

# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

# Define mixing parameter
mixing_method = universe_params["mixing"]
alphas_d = {
    'work':mixing_method['param_alpha'],
    'transport':mixing_method['param_alpha'],
    'school':mixing_method['param_alpha'],
    'other':mixing_method['param_alpha'],
    'leisure':mixing_method['param_alpha'],
    'home':mixing_method['param_alpha'],
}
fast_mixing_method = {
	"name":"mult",
	"param_alpha":alphas_d,
	"param_beta":alphas_d,
}


def ICU_admissions_trigger_policy(experiment_params, thresholds, activity_levels):


	result_policy = {
		"lockdown_heuristic":"ICU_admissions_trigger",
		"groups":groups,
		"experiment_params":{
			"delta_schooling":experiment_params["delta_schooling"],
			"xi":experiment_params["xi"],
			"icus":experiment_params["icus"],
			"n_a_tests":experiment_params["tests"][1],
			"n_m_tests":experiment_params["tests"][0],
			"start_day":start_day,
			"T":simulation_params['time_periods'],
			"eta":experiment_params["eta"],
			"test_freq":simulation_params["days"],
			"policy_freq":quar_freq,
			"end_days":14,			
		},
		"testing_heuristic":experiment_params["testing"],
	}
	result_policy["filename"] = "%s/xi-%d_icus-%d_testing-%s_natests-%d_nmtests-%d_T-%d_startday-%d_groups-%s_dschool-%f_eta-%f_freq-%d-%d"%(
		result_policy["lockdown_heuristic"],
		result_policy["experiment_params"]["xi"],
		result_policy["experiment_params"]["icus"],
		result_policy["testing_heuristic"],
		result_policy["experiment_params"]["n_a_tests"],
		result_policy["experiment_params"]["n_m_tests"],
		result_policy["experiment_params"]["T"],
		result_policy["experiment_params"]["start_day"],
		result_policy["groups"],
		result_policy["experiment_params"]["delta_schooling"],
		experiment_params["eta"],
		result_policy["experiment_params"]["test_freq"],
		result_policy["experiment_params"]["policy_freq"],
	)

	alpha_H = activity_levels["high_activity"]
	alpha_L = activity_levels["low_activity"]
	high_activity_policy = {
		"home": 1.0,
		"leisure": alpha_H,
		"other": alpha_H,
		"school": alpha_H,
		"transport": alpha_H,
		"work": alpha_H
		}
	low_activity_policy = {
		"home": 1.0,
		"leisure": alpha_L,
		"other": alpha_L,
		"school": alpha_L,
		"transport": alpha_L,
		"work": alpha_L
		}



	l_policy = []
	a_tests_policy = []
	m_tests_policy = []


	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day, experiment_params["eta"], extra_data = True)

	# Initialize alpha with high level for all groups and activities
	alpha = {ag:deepcopy(high_activity_policy) for ag in age_groups}
	sevendays_newICU = []

	for t in range(simulation_params['time_periods']):
		# Store the alpha (i.e., lockdown decisions) used previously
		alpha_previous = alpha
		
		# Recover the state before taking a time step, so as to get the "I" and "Iss" states before taking the next step. 
		# These will yield the new flow into ICU in the next step.
		state = dynModel.get_state(t)
		
		# Find new ICU admissions over last 7 days
		newICU = np.sum([dynModel.groups[ag].parameters['mu']*(dynModel.groups[ag].parameters['p_ICU']*state[ag]["I"] + ((dynModel.groups[ag].parameters['p_ICU'])/(dynModel.groups[ag].parameters['p_H']+dynModel.groups[ag].parameters['p_ICU']))*state[ag]["Iss"]) for ag in age_groups])
		sevendays_newICU.append(newICU)
		# Drop first entry if larger than 7
		if len(sevendays_newICU)>7:
			sevendays_newICU = sevendays_newICU[1:8]
		sevendays_avg_newICU = np.sum(sevendays_newICU)/len(sevendays_newICU)
		
		# Take a time step. The ICU utilization rate will be calculated using the resulting ICU state.
		result = dynModel.take_time_step(m_tests, a_tests, alpha)
		new_state = result["state"]
		
		# Find ICU utilization rate
		icu_utilization_rate = np.sum([new_state[ag]["ICU"] for ag in age_groups])/experiment_params["icus"]
		
		
		if sevendays_avg_newICU > thresholds["sevenday_avg_newICU"]:
			alpha = {ag:deepcopy(low_activity_policy) for ag in age_groups}
		elif sevendays_avg_newICU <= thresholds["sevenday_avg_newICU"] and icu_utilization_rate<=thresholds["icu_utilization_rate"]:
			alpha = {ag:deepcopy(high_activity_policy) for ag in age_groups}
		else: 
			alpha = alpha_previous


		l_policy.append(deepcopy(alpha))
		a_tests_policy.append(deepcopy(a_tests))
		m_tests_policy.append(deepcopy(m_tests))


	end_alphas, end_a_tests, end_m_tests = dynModel.take_end_steps()

	l_policy += end_alphas
	a_tests_policy += end_a_tests
	m_tests_policy += end_m_tests

	result_policy.update({
		"results":{
			"economics_value":float(dynModel.get_total_economic_value()),
			"deaths":float(dynModel.get_total_deaths()),
			"reward":float(dynModel.get_total_reward()),
		},
		"policy":l_policy,
		"a_tests":a_tests_policy,
		"m_tests":m_tests_policy,
	})


	return result_policy

def ICU_admissions_trigger_policy_fast(experiment_params, thresholds, activity_levels):

	alpha_H = activity_levels["high_activity"]
	alpha_L = activity_levels["low_activity"]
	high_activity_policy = {
		"home": 1.0,
		"leisure": alpha_H,
		"other": alpha_H,
		"school": alpha_H,
		"transport": alpha_H,
		"work": alpha_H
		}
	low_activity_policy = {
		"home": 1.0,
		"leisure": alpha_L,
		"other": alpha_L,
		"school": alpha_L,
		"transport": alpha_L,
		"work": alpha_L
		}

	high_activity_matrix = np.zeros((len(age_groups),len(activities)))
	low_activity_matrix = np.zeros((len(age_groups),len(activities)))
	for i,ag in enumerate(age_groups):
		for j,act in enumerate(activities):
			high_activity_matrix[i,j] = high_activity_policy[act]
			low_activity_matrix[i,j] = low_activity_policy[act]

	state = initial_state
	total_reward = 0
	total_deaths = 0
	total_econ = 0
	current_activity = "high"
	update_contacts = True
	current_lockdowns = high_activity_matrix
	sevendays_newICU = []

	for t in range(simulation_params['time_periods']):

		state, econs = fastModel.take_time_step(
			state, 
			m_tests_vec, 
			a_tests_vec,
			current_lockdowns,
			t,
			"spc",
			update_contacts = update_contacts, 
			B_H = False, 
			B_ICU = False,
			B_H_perc = False,
			B_ICU_perc = False,
		)
		total_reward += econs['reward']
		total_deaths += econs['deaths']
		total_econ += econs['economic_value']

		newICU = np.sum(fastModel.get_flow_ICU())
		sevendays_newICU.append(newICU)

		if len(sevendays_newICU)>7:
			sevendays_newICU = sevendays_newICU[1:8]
		sevendays_avg_newICU = np.sum(sevendays_newICU)/len(sevendays_newICU)

		icu_utilization_rate = np.sum(state[:,cont.index("ICU")])/experiment_params["icus"]

		previous_lockdowns = current_lockdowns
		previous_activity = current_activity

		if sevendays_avg_newICU > thresholds["sevenday_avg_newICU"]:
			current_lockdowns = low_activity_matrix
			current_activity = "low"
		elif sevendays_avg_newICU <= thresholds["sevenday_avg_newICU"] and icu_utilization_rate<=thresholds["icu_utilization_rate"]:
			current_lockdowns = high_activity_matrix
			current_activity = "high"
		else: 
			current_lockdowns = previous_lockdowns
			current_activity = previous_activity

		if current_activity == previous_activity:
			update_contacts = False
		else:
			update_contacts = True

	for t in range(fastModel.END_DAYS):
		state, econs = fastModel.take_end_step(state,"spc")
		total_reward += econs['reward']
		total_deaths += econs['deaths']
		total_econ += econs['economic_value']


	result = {
		"economics_value":total_econ,
		"deaths":total_deaths,
		"reward":total_reward,	
		"thresholds":thresholds,
		"activity_levels":activity_levels,
	}

	return result

partial_results = []
all_results = []
for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					for eta in params_to_try["eta"]:
						experiment_params = {
							'delta_schooling':delta,
							'xi':xi,
							'icus':icus,
							'testing':testing,
							'tests':tests,
							'eta':eta,
						}

						counter = 0
						best_reward = -float("inf")
						best_result = 0

						# Create dynamical method
						fastModel = FastDynamicalModel(universe_params, econ_params, experiment_params, 1, fast_mixing_method, simulation_params['time_periods'], start_day, experiment_params["eta"])
						initial_state = state_to_matrix(initialization)
						if experiment_params["testing"] == "homogeneous":
							m_tests = {ag:experiment_params["tests"][0]/len(age_groups) for ag in age_groups}
							a_tests = {ag:experiment_params["tests"][1]/len(age_groups) for ag in age_groups}

						m_tests_vec = dict_to_vector(m_tests)
						a_tests_vec = dict_to_vector(a_tests)

						total_iterations = (
							(len(thresholds_to_try["icu_utilization_rate"])-1)/2*
							len(thresholds_to_try["sevenday_avg_newICU"])*
							len(lockdowns_to_try["high_activity"])*
							len(lockdowns_to_try["low_activity"])
						)

						for icu_util_rate_t in thresholds_to_try["icu_utilization_rate"]:
							for sevenday_avg_newICU_t in thresholds_to_try["sevenday_avg_newICU"]:
								for act_H_t in lockdowns_to_try["high_activity"]:
									for act_L_t in lockdowns_to_try["low_activity"]:
										if act_L_t<act_H_t:
											thresholds = {
												"icu_utilization_rate":icu_util_rate_t,
												"sevenday_avg_newICU":sevenday_avg_newICU_t,
											}
											activity_levels = {
												"high_activity":act_H_t,
												"low_activity":act_L_t,
											}
									
											partial_result = ICU_admissions_trigger_policy_fast(experiment_params, thresholds, activity_levels)

											partial_results.append(partial_result)

											if partial_result["reward"]>best_reward:
												best_reward = partial_result["reward"]
												best_result = partial_result
												print(best_result)

											print(counter/total_iterations)
											counter+=1

						result_policy = ICU_admissions_trigger_policy(experiment_params, best_result["thresholds"], best_result["activity_levels"])
						result_policy["thresholds"] = best_result["thresholds"]
						result_policy["activity_levels"] = best_result["activity_levels"]
						print(result_policy.keys())
						all_results.append(result_policy)



for r in all_results:
	fn =  "results/"+r["filename"]+".yaml"
	print(fn)
	with open(fn, 'w') as file:
		yaml.dump(r, file)



