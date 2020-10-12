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
from copy import deepcopy


# Parameters to try
with open("../parameters/run_params.yaml") as file:
	run_params = yaml.load(file, Loader=yaml.FullLoader)

params_to_try = run_params["params_to_try"]
groups = run_params["groups"]

proportions = {'age_group_0_9': 0.12999753718396828, 'age_group_10_19': 0.1260199381062682, 'age_group_20_29': 0.13462273540296374, 'age_group_30_39': 0.1432185965976917, 'age_group_40_49': 0.13619350895266272, 'age_group_50_59': 0.1252867882416867, 'age_group_60_69': 0.09586005862219948, 'age_group_70_79': 0.06449748382900194, 'age_group_80_plus': 0.044303353063557066}

death_prob_all = {
	"age_group_0_9":0.002*0.006,
	"age_group_10_19":0.002*0.006,
	"age_group_20_29":0.006*0.011,
	"age_group_30_39":0.013*0.019,
	"age_group_40_49":0.017*0.033,
	"age_group_50_59":0.035*0.065,
	"age_group_60_69":0.071*0.126,
	"age_group_70_79":0.113*0.21,
	"age_group_80_plus":0.32*0.316,
}

death_prob_one = {
	"all_age_groups":sum([death_prob_all[ag]*proportions[ag] for ag in death_prob_all])
}

if groups == "all":
	death_prob = death_prob_all
elif groups == "one":
	death_prob = death_prob_one


# full_open_policy = {
# 	"home": 1.0,
# 	"leisure": 1.0,
# 	"other": 1.0,
# 	"school": 1.0,
# 	"transport": 1.0,
# 	"work": 1.0
# }	


# Global variables
simulation_params = {
	'dt':1.0,
	'days': 30.0,
	# 'days': 180.0,
	'region': "fitted",
	'heuristic': 'benchmark',
	'mixing_method': {'name': 'multi'}
}
# simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

if groups == "all":
	age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']

	# Read group parameters
	with open("../parameters/fitted.yaml") as file:
	    universe_params = yaml.load(file, Loader=yaml.FullLoader)

	# Read initialization
	with open("../initialization/60days.yaml") as file:
		initialization = yaml.load(file, Loader=yaml.FullLoader)
		start_day = 60

	# Read econ parameters
	with open("../parameters/econ.yaml") as file:
		econ_params = yaml.load(file, Loader=yaml.FullLoader)

elif groups == "one":
	age_groups = ["all_age_groups"]

	# Read group parameters
	with open("../parameters/one_group_fitted.yaml") as file:
	    universe_params = yaml.load(file, Loader=yaml.FullLoader)

	# Read initialization
	with open("../initialization/60days_one_group.yaml") as file:
		initialization = yaml.load(file, Loader=yaml.FullLoader)
		start_day = 60

	# Read econ parameters
	with open("../parameters/one_group_econ.yaml") as file:
		econ_params = yaml.load(file, Loader=yaml.FullLoader)

# Read lower bounds
# with open("../lower_bounds/fitted.yaml") as file:
# 	lower_bounds = yaml.load(file, Loader=yaml.FullLoader)




# cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
#        'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]
# activities = ['home','leisure','other','school','transport','work']
# rel_activities = ['leisure','other','school','transport','work']


# age_groups = ['age_group_30_39']

# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))




# Define mixing parameter
mixing_method = universe_params["mixing"]

# Load gov policy
with open("../policies/fitted.yaml") as file:
	gov_policy = yaml.load(file, Loader=yaml.FullLoader)
for i,p in enumerate(gov_policy):
	if p["days_from_lockdown"] == 0:
		start_lockdown = i
		break
for i,p in enumerate(gov_policy):
	del p['date']
	del p['days_from_lockdown']





# Some basic policies

full_lockdown_policy = gov_policy[start_lockdown]



def run_government_policy(experiment_params):

	alphas_vec = []
	l_policy = []
	a_tests_policy = []
	m_tests_policy = []

	for t in range(simulation_params['time_periods']):
		index = t+start_day
		if t+start_day >= len(gov_policy):
			alphas_vec.append({ag:deepcopy(gov_policy[-1]) for ag in age_groups})
		else:
			alphas_vec.append({ag:deepcopy(gov_policy[t+start_day]) for ag in age_groups})

	# Create dynamical method
	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day, experiment_params["eta"])
	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"][0]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"][1]/len(age_groups) for ag in age_groups}

	for t in range(simulation_params['time_periods']):
		dynModel.take_time_step(m_tests, a_tests, alphas_vec[t])
		l_policy.append(deepcopy(alphas_vec[t]))
		a_tests_policy.append(deepcopy(a_tests))
		m_tests_policy.append(deepcopy(m_tests))

	# Finish with the last steps
	end_alphas, end_a_tests, end_m_tests = dynModel.take_end_steps()

	l_policy += end_alphas
	a_tests_policy += end_a_tests
	m_tests_policy += end_m_tests

	result = {
		"lockdown_heuristic":"real",
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
			"test_freq":1,
			"policy_freq":1,
			"end_days":14,
		},
		"testing_heuristic":experiment_params["testing"],
		"results":{
			"economics_value":float(dynModel.get_total_economic_value()),
			"deaths":float(dynModel.get_total_deaths()),
			"reward":float(dynModel.get_total_reward()),
		},
		"policy":l_policy,
		"a_tests":a_tests_policy,
		"m_tests":m_tests_policy,
	}

	result["filename"] = "%s/xi-%d_icus-%d_testing-%s_natests-%d_nmtests-%d_T-%d_startday-%d_groups-%s_dschool-%f_eta-%f_freq-%d-%d"%(
		result["lockdown_heuristic"],
		result["experiment_params"]["xi"],
		result["experiment_params"]["icus"],
		result["testing_heuristic"],
		result["experiment_params"]["n_a_tests"],
		result["experiment_params"]["n_m_tests"],
		result["experiment_params"]["T"],
		result["experiment_params"]["start_day"],
		result["groups"],
		result["experiment_params"]["delta_schooling"],
		result["experiment_params"]["eta"],
		result["experiment_params"]["test_freq"],
		result["experiment_params"]["policy_freq"],
	)


	return result



def run_full_lockdown(experiment_params):

	ag_alpha = {
		"home": 1.0,
	    "leisure": 0,
	    "other": 0,
	    "school": 0,
	    "transport": 0,
	    "work": 0,
	}

	alpha = {
		ag:deepcopy(ag_alpha) for ag in age_groups
	}

	l_policy = []
	a_tests_policy = []
	m_tests_policy = []

	# Create dynamical method
	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day, experiment_params["eta"])
	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"][0]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"][1]/len(age_groups) for ag in age_groups}

	for t in range(simulation_params['time_periods']):
		dynModel.take_time_step(m_tests, a_tests, alpha)
		l_policy.append(deepcopy(alpha))
		a_tests_policy.append(deepcopy(a_tests))
		m_tests_policy.append(deepcopy(m_tests))

	end_alphas, end_a_tests, end_m_tests = dynModel.take_end_steps()

	l_policy += end_alphas
	a_tests_policy += end_a_tests
	m_tests_policy += end_m_tests

	result = {
		"lockdown_heuristic":"full_lockdown",
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
			"test_freq":1,
			"policy_freq":1,
			"end_days":14,
		},
		"testing_heuristic":experiment_params["testing"],
		"results":{
			"economics_value":float(dynModel.get_total_economic_value()),
			"deaths":float(dynModel.get_total_deaths()),
			"reward":float(dynModel.get_total_reward()),
		},
		"policy":l_policy,
		"a_tests":a_tests_policy,
		"m_tests":m_tests_policy,
	}

	result["filename"] = "%s/xi-%d_icus-%d_testing-%s_natests-%d_nmtests-%d_T-%d_startday-%d_groups-%s_dschool-%f_eta-%f_freq-%d-%d"%(
		result["lockdown_heuristic"],
		result["experiment_params"]["xi"],
		result["experiment_params"]["icus"],
		result["testing_heuristic"],
		result["experiment_params"]["n_a_tests"],
		result["experiment_params"]["n_m_tests"],
		result["experiment_params"]["T"],
		result["experiment_params"]["start_day"],
		result["groups"],
		result["experiment_params"]["delta_schooling"],
		result["experiment_params"]["eta"],
		result["experiment_params"]["test_freq"],
		result["experiment_params"]["policy_freq"],
	)






	return result

def run_open(experiment_params):
	ag_alpha = {
		"home": 1.0,
	    "leisure": 1.0,
	    "other": 1.0,
	    "school": 1.0,
	    "transport": 1.0,
	    "work": 1.0,
	}

	alpha = {
		ag:deepcopy(ag_alpha) for ag in age_groups
	}

	l_policy = []
	a_tests_policy = []
	m_tests_policy = []

	# Create dynamical method
	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day, experiment_params["eta"])
	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"][0]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"][1]/len(age_groups) for ag in age_groups}

	for t in range(simulation_params['time_periods']):
		dynModel.take_time_step(m_tests, a_tests, alpha)
		l_policy.append(deepcopy(alpha))
		a_tests_policy.append(deepcopy(a_tests))
		m_tests_policy.append(deepcopy(m_tests))

	end_alphas, end_a_tests, end_m_tests = dynModel.take_end_steps()

	l_policy += end_alphas
	a_tests_policy += end_a_tests
	m_tests_policy += end_m_tests


	result = {
		"lockdown_heuristic":"full_open",
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
			"test_freq":1,
			"policy_freq":1,
			"end_days":14,
		},
		"testing_heuristic":experiment_params["testing"],
		"results":{
			"economics_value":float(dynModel.get_total_economic_value()),
			"deaths":float(dynModel.get_total_deaths()),
			"reward":float(dynModel.get_total_reward()),
		},
		"policy":l_policy,
		"a_tests":a_tests_policy,
		"m_tests":m_tests_policy,
	}

	result["filename"] = "%s/xi-%d_icus-%d_testing-%s_natests-%d_nmtests-%d_T-%d_startday-%d_groups-%s_dschool-%f_eta-%f_freq-%d-%d"%(
		result["lockdown_heuristic"],
		result["experiment_params"]["xi"],
		result["experiment_params"]["icus"],
		result["testing_heuristic"],
		result["experiment_params"]["n_a_tests"],
		result["experiment_params"]["n_m_tests"],
		result["experiment_params"]["T"],
		result["experiment_params"]["start_day"],
		result["groups"],
		result["experiment_params"]["delta_schooling"],
		result["experiment_params"]["eta"],
		result["experiment_params"]["test_freq"],
		result["experiment_params"]["policy_freq"],
	)


	return result



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

						result_real = run_government_policy(experiment_params)
						result_closed = run_full_lockdown(experiment_params)
						result_open = run_open(experiment_params)

						all_results.append(result_real)
						all_results.append(result_closed)
						all_results.append(result_open)




for r in all_results:
	fn =  "results/"+r["filename"]+".yaml"
	print(fn)
	with open(fn, 'w') as file:
		yaml.dump(r, file)


#pd.DataFrame(all_results).to_excel(f"simulations-{simulation_params['days']}-days.xlsx")
