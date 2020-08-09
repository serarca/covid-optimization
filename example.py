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
import pickle
import numpy as np

# Global variables
simulation_params = {
	'dt':1.0,
	'days': 180,
	'region': "fitted",
	'heuristic': 'benchmark',
	'mixing_method': {'name': 'multi'}
}
age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']

# age_groups = ['age_group_30_39']

# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))


# Read group parameters
with open("./parameters/fitted.yaml") as file:
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("./initialization/61days.yaml") as file:
	initialization = yaml.load(file, Loader=yaml.FullLoader)

# Read econ parameters
with open("./parameters/econ.yaml") as file:
	econ_params = yaml.load(file, Loader=yaml.FullLoader)

# Define mixing parameter
mixing_method = universe_params["mixing"]


# Parameters to try
params_to_try = {
	"delta_schooling":0.5,
	"xi":0 * 37199.03,
	"icus":2000,
	"tests":30000,
	"testing":"homogeneous"
}

# We start by benchmarking the policy implemented by the government
# Read econ parameters
with open("./policies/fitted.yaml") as file:
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
		del gov_policy[t+start_lockdown_day]['lockdown']
		alphas_vec.append({ag:gov_policy[t+start_lockdown_day] for ag in age_groups})


delta = params_to_try["delta_schooling"]
xi = params_to_try["xi"]
icus = params_to_try["icus"]
tests = params_to_try["tests"]
testing = params_to_try["testing"]

experiment_params = {
	'delta_schooling':delta,
	'xi':xi,
	'icus':icus,
}

# Create dynamical method
dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)

# Homogenous testing
m_tests = {ag:tests/len(age_groups) for ag in age_groups}
a_tests = {ag:tests/len(age_groups) for ag in age_groups}

for t in range(simulation_params['time_periods']):
	dynModel.take_time_step(m_tests, a_tests, alphas_vec[t])

results = {
	"heuristic":"real",
	"delta_schooling":delta,
	"xi":xi,
	"icus":icus,
	"tests":tests,
	"testing":testing,
	"economics_value":dynModel.get_total_economic_value(),
	"deaths":dynModel.get_total_deaths(),
	"reward":dynModel.get_total_reward(),
}










