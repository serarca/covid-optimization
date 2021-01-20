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
from os import listdir
from os.path import isfile, join


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


cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]
activities = ['home','leisure','other','school','transport','work']
rel_activities = ['leisure','other','school','transport','work']


# age_groups = ['age_group_30_39']

# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

# Read files
folder = "../benchmarks/results/dynamic_gradient/"
files = [f for f in listdir(folder) if (isfile(join(folder, f)) and f[0:2]=="xi")]


files_to_process = []
for i,f in enumerate(files):
	
	keys = f.split("_")
	if (
		("dschool-0.500000" in keys) and
		("eta-0.200000" in keys) and
		("natests-0" in keys) and
		("nmtests-0" in keys)
	):
		files_to_process.append(f)


print(len(files_to_process))
features = []
for i,f in enumerate(files_to_process):
	print(i/len(files_to_process))
	with open(folder+f) as file:
		file_result = yaml.load(file, Loader=yaml.FullLoader)

	experiment_params = file_result["experiment_params"]
	mixing_method = universe_params["mixing"]

	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day, experiment_params["eta"], extra_data = True)

	for t in range(simulation_params['time_periods']):
		state = dynModel.get_state(t)
		result = dynModel.take_time_step(file_result['m_tests'][t], file_result['a_tests'][t], file_result['policy'][t])

		econ_gradients = dynModel.econ_gradients

		# Extract the features
		for act in rel_activities:
			for ag in age_groups:
				features.append({
					"act":act,
					"ag":ag,
					"lockdown":file_result['policy'][t][ag][act],
					"econ_gradient":econ_gradients[act][ag]/state[ag]["N"],
					"I_perc":np.sum([state[other_group]["I"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
					"R_perc":np.sum([state[other_group]["R"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
					"H_perc":np.sum([state[other_group]["H"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
					"ICU_utilization":np.sum([state[other_group]["ICU"] for other_group in age_groups])/experiment_params["icus"],
					"xi+cost_death":experiment_params["xi"]+econ_params["econ_cost_death"][ag],
					"cost_death":econ_params["econ_cost_death"][ag],
					"xi":experiment_params["xi"],
					"contacts_received":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group] for other_group in age_groups]),
					"contacts_received(deaths)":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group]*universe_params["seir-groups"][ag]['parameters']['lambda_H_D']*14.55 for other_group in age_groups]),
					"infections_received":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"]) for other_group in age_groups]),
					"deaths_received":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"])*universe_params["seir-groups"][ag]['parameters']['lambda_H_D']*14.55 for other_group in age_groups]),
					"deaths_econ_received":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"])*universe_params["seir-groups"][ag]['parameters']['lambda_H_D']*14.55*(experiment_params["xi"]+econ_params["econ_cost_death"][ag]) for other_group in age_groups]),
					"contacts_given":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag] for other_group in age_groups]),
					"contacts_given(deaths)":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag]*universe_params["seir-groups"][other_group]['parameters']['lambda_H_D']*14.55 for other_group in age_groups]),
					"infections_given":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"]) for other_group in age_groups]),
					"deaths_given":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"])*universe_params["seir-groups"][other_group]['parameters']['lambda_H_D']*14.55 for other_group in age_groups]),
					"deaths_econ_given":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"])*universe_params["seir-groups"][other_group]['parameters']['lambda_H_D']*14.55*(experiment_params["xi"]+econ_params["econ_cost_death"][other_group]) for other_group in age_groups]),
					"death_prob":universe_params["seir-groups"][ag]['parameters']['lambda_H_D']*14.55,
					"t":t,
				})


pd.DataFrame(features).to_csv("dataset.csv")







