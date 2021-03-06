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
import time

from group import SEIR_group, DynamicalModel
import math
import pprint
import pandas as pd
import pickle
import numpy as np
from fast_group import FastDynamicalModel
from aux import *
from scipy.optimize import Bounds,minimize,LinearConstraint
from random import sample 



groups = "one"
experiment_params = {
	'delta_schooling':0.5,
	'xi':1115970.9,
	'icus':3000,
	'testing':"homogeneous",
	'tests':0,
	'eta':0.1,
}

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
	population = sum([initialization["all_age_groups"][cont] for cont in initialization["all_age_groups"] if cont!="N"])


	infected_perc = sample([0.1,0.01,0.001,0.0001,0.00001,0.000001],1)[0]*np.random.random()
	n_icu = sample([1.0,0.1,0.01,0],1)[0]*experiment_params["icus"] * np.random.random()
	recovered_perc = sample([0.9,0.1,0.001,0.0001,0],1)[0]*np.random.random()


	s_perc = 1.0 - recovered_perc - infected_perc
	
	initialization = {'all_age_groups': 
		{'D': 0, 
		'E': infected_perc/2*(population-5*n_icu), 
		'H': 4*n_icu, 
		'I': infected_perc/2*(population-5*n_icu), 
		'ICU': n_icu, 
		'Ia': 0.0, 
		'Ims': 0.0, 
		'Ips': 0.0, 
		'Iss': 0.0, 
		'N': population, 
		'R': recovered_perc*(population-5*n_icu), 
		'Rq': 0,
		'S': s_perc*(population-5*n_icu)}
	}

	assert(sum([initialization['all_age_groups'][cont] for cont in initialization['all_age_groups'] if cont!="N"]) == population)

	# Read econ parameters
	with open("../parameters/one_group_econ.yaml") as file:
		econ_params = yaml.load(file, Loader=yaml.FullLoader)

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







def gradient_descent(experiment_params, quar_freq, plot=False):


	result = {
		"lockdown_heuristic":"dynamic_gradient",
		"groups":groups,
		"experiment_params":{
			"delta_schooling":experiment_params["delta_schooling"],
			"xi":experiment_params["xi"],
			"icus":experiment_params["icus"],
			"n_a_tests":experiment_params["tests"],
			"n_m_tests":experiment_params["tests"],
			"start_day":start_day,
			"T":simulation_params['time_periods'],
			"eta":econ_params["employment_params"]["eta"],
			"test_freq":simulation_params["days"],
			"policy_freq":quar_freq,
			"end_days":14,			
		},
		"testing_heuristic":experiment_params["testing"],
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



	intervention_times = [t*quar_freq for t in range(int((simulation_params['days']-1)/quar_freq)+1)]

	x0 = np.random.random(len(intervention_times)*len(age_groups)*len(rel_activities))

	# Create dynamical model
	fastModel = FastDynamicalModel(universe_params, econ_params, experiment_params, simulation_params['dt'], mixing_method, simulation_params['time_periods'], start_day, experiment_params["eta"])
	initial_state = state_to_matrix(initialization)

	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}

	m_tests_vec = dict_to_vector(m_tests)
	a_tests_vec = dict_to_vector(a_tests)


	def simulate(x):
		# Extract vector components
		x_lockdown = np.reshape(
			x,
			(len(intervention_times),len(age_groups),len(rel_activities))
		)


		state = initial_state
		total_reward = 0
		total_deaths = 0
		total_econ = 0
		for t in range(simulation_params['time_periods']):
			if t in intervention_times:
				update_contacts = True
			else:
				update_contacts = False
			
			# Add home activity to lockdown
			x_lockdown_all = np.concatenate((np.array([[1.0]]*len(age_groups)),x_lockdown[int(t/quar_freq),:,:]), axis=1)


			state, econs = fastModel.take_time_step(
				state, 
				m_tests_vec, 
				a_tests_vec,
				x_lockdown_all,
				t,
				update_contacts = update_contacts, 
				B_H = False, 
				B_ICU = False,
				B_H_perc = False,
				B_ICU_perc = False,
			)
			total_reward += econs['reward']
			total_deaths += econs['deaths']
			total_econ += econs['economic_value']

		for t in range(fastModel.END_DAYS):
			state, econs = fastModel.take_end_step(state)
			total_reward += econs['reward']
			total_deaths += econs['deaths']
			total_econ += econs['economic_value']




		print(total_reward, total_deaths, total_econ)
		return -total_reward

	lower_bounds_matrix = np.zeros((len(intervention_times),len(age_groups),len(rel_activities)))
	for i,act in enumerate(rel_activities):
		lower_bounds_matrix[:,:,i] += lower_bounds[act]




	
	full_bounds = Bounds(
		lower_bounds_matrix.flatten(),
		np.zeros(len(intervention_times)*len(age_groups)*len(rel_activities)) + 1.0
	)

	result_lockdown = minimize(simulate, x0, method='L-BFGS-B',bounds=full_bounds,options={'eps':1e-1,'maxfun':700000})

	x_lockdown = np.reshape(
		result_lockdown.x,
		(len(intervention_times),len(age_groups),len(rel_activities))
	)

	alpha = matrix_to_alphas(x_lockdown, quar_freq)


	l_policy = []
	a_tests_policy = []
	m_tests_policy = []
	states = [initialization]
	# Create dynamical method
	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day, experiment_params["eta"], extra_data = True)
	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}

	for t in range(simulation_params['time_periods']):
		r = dynModel.take_time_step(m_tests, a_tests, alpha[t])
		l_policy.append(deepcopy(alpha[t]))
		a_tests_policy.append(deepcopy(a_tests))
		m_tests_policy.append(deepcopy(m_tests))
		states.append(r["state"])


	end_alphas, end_a_tests, end_m_tests = dynModel.take_end_steps()

	l_policy += end_alphas
	a_tests_policy += end_a_tests
	m_tests_policy += end_m_tests

	result.update({
		"results":{
			"economics_value":float(dynModel.get_total_economic_value()),
			"deaths":float(dynModel.get_total_deaths()),
			"reward":float(dynModel.get_total_reward()),
		},
		"policy":l_policy,
		"a_tests":a_tests_policy,
		"m_tests":m_tests_policy,
		"states":states,
	})

	return result



result_gradient = gradient_descent(experiment_params, 14)

with open("runs/run_%s.yaml"%str(time.time()), 'w') as file:
		yaml.dump(result_gradient, file)



