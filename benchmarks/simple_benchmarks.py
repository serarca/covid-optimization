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
sys.path.insert(0, current_dir+"/heuristics/LP-Models")
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from group import SEIR_group, DynamicalModel
import math
import pprint
import pandas as pd


# Global variables
simulation_params = {
	'dt':1.0,
	'days': 90.0,
	'region': "Ile-de-France",
}
age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']

# age_groups = ['age_group_30_39']

# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))


# Read group parameters
with open("../parameters/fitted.yaml") as file:
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("../initialization/fitted.yaml") as file:
	initialization = yaml.load(file, Loader=yaml.FullLoader)

# Read econ parameters
with open("../parameters/econ.yaml") as file:
	econ_params = yaml.load(file, Loader=yaml.FullLoader)

# Define mixing parameter
mixing_method = universe_params["mixing"]


# Parameters to try
params_to_try = {
	"delta_schooling":[0.5],
	"xi":[1 * 37199.03, 30 * 37199.03],
	"icus":[2000,2500],
	"tests":[0,30000],
	"testing":["homogeneous"]+age_groups
}
results = []


# We start by benchmarking the policy implemented by the government
# Read econ parameters
with open("../policies/fitted.yaml") as file:
	gov_policy = yaml.load(file, Loader=yaml.FullLoader)

alphas_vec = []
start_lockdown_day = [i for i,d in enumerate(gov_policy) if d['days_from_lockdown']==0][0]


for t in range(simulation_params['time_periods']):
	index = t+start_lockdown_day
	if t+start_lockdown_day >= len(gov_policy):
		alphas_vec.append({ag:gov_policy[-1] for ag in age_groups})
	else:
		del gov_policy[t+start_lockdown_day]['date']
		del gov_policy[t+start_lockdown_day]['days_from_lockdown']
		alphas_vec.append({ag:gov_policy[t+start_lockdown_day] for ag in age_groups})


for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					experiment_params = {
						'delta_schooling':delta,
						'xi':xi,
						'icus':icus,
					}
					# Create dynamical method
					dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
					if testing == "homogeneous":
						m_tests = {ag:tests/len(age_groups) for ag in age_groups}
						a_tests = {ag:tests/len(age_groups) for ag in age_groups}
					elif testing in age_groups:
						m_tests = {ag:tests if ag==testing else 0 for ag in age_groups}
						a_tests = {ag:tests if ag==testing else 0 for ag in age_groups}

					for t in range(simulation_params['time_periods']):
						dynModel.take_time_step(m_tests, a_tests, alphas_vec[t])

					results.append({
						"heuristic":"real",
						"delta_schooling":delta,
						"xi":xi,
						"icus":icus,
						"tests":tests,
						"testing":testing,
						"economics_value":dynModel.get_total_economic_value(),
						"deaths":dynModel.get_total_deaths(),
						"reward":dynModel.get_total_reward(),
					})



# Now we benchmark government full lockdown
alphas={ag:gov_policy[start_lockdown_day] for ag in age_groups}


for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					experiment_params = {
						'delta_schooling':delta,
						'xi':xi,
						'icus':icus,
					}
					# Create dynamical method
					dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
					if testing == "homogeneous":
						m_tests = {ag:tests/len(age_groups) for ag in age_groups}
						a_tests = {ag:tests/len(age_groups) for ag in age_groups}
					elif testing in age_groups:
						m_tests = {ag:tests if ag==testing else 0 for ag in age_groups}
						a_tests = {ag:tests if ag==testing else 0 for ag in age_groups}

					for t in range(simulation_params['time_periods']):
						dynModel.take_time_step(m_tests, a_tests, alphas)

					results.append({
						"heuristic":"government full lockdown",
						"delta_schooling":delta,
						"xi":xi,
						"icus":icus,
						"tests":tests,
						"testing":testing,
						"economics_value":dynModel.get_total_economic_value(),
						"deaths":dynModel.get_total_deaths(),
						"reward":dynModel.get_total_reward(),
					})



# Now we benchmark zero full lockdown
alphas={ag:{
	"home": 1.0,
	"leisure": 0.0,
	"other": 0.0,
	"school": 0.0,
	"transport": 0.0,
	"work": 0.0
} for ag in age_groups}


for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					experiment_params = {
						'delta_schooling':delta,
						'xi':xi,
						'icus':icus,
					}
					# Create dynamical method
					dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
					if testing == "homogeneous":
						m_tests = {ag:tests/len(age_groups) for ag in age_groups}
						a_tests = {ag:tests/len(age_groups) for ag in age_groups}
					elif testing in age_groups:
						m_tests = {ag:tests if ag==testing else 0 for ag in age_groups}
						a_tests = {ag:tests if ag==testing else 0 for ag in age_groups}

					for t in range(simulation_params['time_periods']):
						dynModel.take_time_step(m_tests, a_tests, alphas)

					results.append({
						"heuristic":"zero full lockdown",
						"delta_schooling":delta,
						"xi":xi,
						"icus":icus,
						"tests":tests,
						"testing":testing,
						"economics_value":dynModel.get_total_economic_value(),
						"deaths":dynModel.get_total_deaths(),
						"reward":dynModel.get_total_reward(),
					})


# Now we benchmark open
alphas={ag:{
	"home": 1.0,
	"leisure": 1.0,
	"other": 1.0,
	"school": 1.0,
	"transport": 1.0,
	"work": 1.0
} for ag in age_groups}


for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					experiment_params = {
						'delta_schooling':delta,
						'xi':xi,
						'icus':icus,
					}
					# Create dynamical method
					dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
					if testing == "homogeneous":
						m_tests = {ag:tests/len(age_groups) for ag in age_groups}
						a_tests = {ag:tests/len(age_groups) for ag in age_groups}
					elif testing in age_groups:
						m_tests = {ag:tests if ag==testing else 0 for ag in age_groups}
						a_tests = {ag:tests if ag==testing else 0 for ag in age_groups}

					for t in range(simulation_params['time_periods']):
						dynModel.take_time_step(m_tests, a_tests, alphas)

					results.append({
						"heuristic":"open",
						"delta_schooling":delta,
						"xi":xi,
						"icus":icus,
						"tests":tests,
						"testing":testing,
						"economics_value":dynModel.get_total_economic_value(),
						"deaths":dynModel.get_total_deaths(),
						"reward":dynModel.get_total_reward(),
					})

pd.DataFrame(results).to_csv("simulations.csv")
