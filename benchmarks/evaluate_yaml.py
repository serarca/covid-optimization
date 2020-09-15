import yaml
from inspect import getsourcefile
import os.path
import os

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




heuristics = ["real","full_open","full_lockdown","constant_gradient","time_gradient","age_group_gradient","linearization_heuristic","linearization_heuristic_Prop_Bouncing"]
all_data = []

for h in heuristics:
	for n in os.listdir("results/%s/"%h):
		if n[0:2] == "xi":

			print(h,n)
			with open("results/%s/%s"%(h,n)) as file:
				result = yaml.load(file, Loader=yaml.UnsafeLoader)

			
			# Read group parameters
			with open("../parameters/fitted.yaml") as file:
			    universe_params_all = yaml.load(file, Loader=yaml.FullLoader)

			# Read econ parameters
			with open("../parameters/econ.yaml") as file:
				econ_params_all = yaml.load(file, Loader=yaml.FullLoader)

			# Read group parameters
			with open("../parameters/one_group_fitted.yaml") as file:
			    universe_params_one = yaml.load(file, Loader=yaml.FullLoader)

			# Read econ parameters
			with open("../parameters/one_group_econ.yaml") as file:
				econ_params_one = yaml.load(file, Loader=yaml.FullLoader)


			start_day = result["experiment_params"]["start_day"]

			if result["groups"] == "one":
				universe_params = universe_params_one
				econ_params = econ_params_one

				with open("../initialization/%ddays_one_group.yaml"%start_day) as file:
					initialization = yaml.load(file, Loader=yaml.FullLoader)

			elif result["groups"] == "all":
				universe_params = universe_params_all
				econ_params = econ_params_all

				with open("../initialization/%ddays.yaml"%start_day) as file:
					initialization = yaml.load(file, Loader=yaml.FullLoader)
			else:
				assert(False)

			experiment_params = result["experiment_params"]
			econ_params["employment_params"]["eta"] = result["experiment_params"]["eta"]
			econ_params["employment_params"]["nu"] = 1- econ_params["employment_params"]["gamma"] - econ_params["employment_params"]["eta"]

			dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, 1, experiment_params["T"], universe_params["mixing"], start_day)

			for t in range(experiment_params["T"]):
				dynModel.take_time_step(result["m_tests"][t], result["a_tests"][t], result["policy"][t])

			dynModel.take_end_steps()

			data = {
				"groups": result["groups"],
				"T": result["experiment_params"]["T"],
				"lock_heuristic":result["lockdown_heuristic"],
				"delta_schooling":result["experiment_params"]["delta_schooling"],
				"xi":result["experiment_params"]["xi"],
				"icus":result["experiment_params"]["icus"],
				"n_a_tests":result["experiment_params"]["n_a_tests"],
				"n_m_tests":result["experiment_params"]["n_m_tests"],
				"test_heuristic":result["testing_heuristic"],
				"eta":result["experiment_params"]["eta"],
				"economics_value":dynModel.get_total_economic_value(),
				"deaths":dynModel.get_total_deaths(),
				"reward":dynModel.get_total_reward(),
				"test_freq":result["experiment_params"]["test_freq"],
			}
			if h in ["linearization_heuristic", "linearization_heuristic_Prop_Bouncing"]:
				data["lock_freq"] = result["experiment_params"]["lockdown_freq"]
				data["test_freq"] = result["experiment_params"]["test_freq"]
			else:
				data["lock_freq"] = result["experiment_params"]["policy_freq"],
			
			all_data.append(data)

pd.DataFrame(all_data).to_excel("results/results.xlsx")

