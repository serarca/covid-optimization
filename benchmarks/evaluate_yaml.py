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

def plot_benchmark(dynModel, result):

    T = dynModel.time_steps

    K_mtest = result["experiment_params"]["n_m_tests"]
    K_atest = result["experiment_params"]["n_a_tests"]

    # Retrieve optimal lockdown decisions
    # Express as dictionary where given an age group, an activity key corresponds to an np.array of length T.
    # That array holds the lockdown decisions for that age group and that activity used in the simulation of dynModel.
    lockdowns_sim = {}
    for n in dynModel.groups:
        lockdowns_sim[n] = {}
        for act in dynModel.lockdown_controls[0][n]:
            lockdowns_sim[n][act] = np.zeros(T)
            for t in range(T):
                lockdowns_sim[n][act][t] = dynModel.lockdown_controls[t][n][act]

    # Retrieve simulated testing decisions
    m_tests_sim = {}
    for n in dynModel.groups:
        m_tests_sim[n] = np.zeros(T)
        for t in range(T):
            m_tests_sim[n][t] = dynModel.m_tests_controls[t][n]

    a_tests_sim = {}
    for n in dynModel.groups:
        a_tests_sim[n] = np.zeros(T)
        for t in range(T):
            a_tests_sim[n][t] = dynModel.a_tests_controls[t][n]


    # # Retrieve simulated bouncing decisions

    B_H_sim = {}
    for n,g in dynModel.groups.items():
        B_H_sim[n] = np.zeros(T-1)
        for t in range(T-1):
            B_H_sim[n][t] = g.B_H[t]

    B_ICU_sim = {}
    for n,g in dynModel.groups.items():
        B_ICU_sim[n] = np.zeros(T-1)
        for t in range(T-1):
            B_ICU_sim[n][t] = g.B_ICU[t]

    print("Deaths through the end of the horizon:", sum([dynModel.deaths[t] for t in range(0,T+1) if t!=0]))
    print("Economic output through the end of the horizon:", sum([dynModel.economic_values[t] for t in range(0,T+1) if t!=0]))


    # Plotting
    time_axis = [i for i in range(T+1)]
    time_axis_controls = [i for i in range(T)]

    groups = dynModel.groups.keys()
    groups = sorted(groups)
    
    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1)
        plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
        plt.title(group)
        plt.legend(loc='upper right')
        plt.ylim(-1,np.max([np.max(dynModel.groups[group].S) for group in groups]))

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups))
        plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
        plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
        plt.legend(loc='upper right')
        plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].E),np.max(dynModel.groups[group].I)) for group in groups]))

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*2)
        plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
        plt.ylim(-1,np.max([np.max(dynModel.groups[group].R) for group in groups]))
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*3)
        plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
        plt.ylim(-1,np.max([np.max(dynModel.groups[group].Rq) for group in groups]))
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*4)
        plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
        plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
        plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
        plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
        plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].Ia),np.max(dynModel.groups[group].Ips),np.max(dynModel.groups[group].Ims),np.max(dynModel.groups[group].Iss)) for group in groups]))
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*5)
        plt.plot(time_axis, dynModel.groups[group].H, label="Hospital")
        plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
        plt.plot(time_axis, dynModel.groups[group].D, label="Dead")
        plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].H),np.max(dynModel.groups[group].ICU),np.max(dynModel.groups[group].D)) for group in groups]))
        plt.legend(loc='upper right')


    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*6)
        plt.plot(time_axis_controls,m_tests_sim[group], label = "M tests")
        plt.plot(time_axis_controls,a_tests_sim[group], label = "A tests")
        # plt.plot(range(0,int(simulation_params['time_periods'])),
        # np.array(re_change_order(m_tests_vec)[group])+max(float(args.m_tests),float(args.a_tests))/100, label="M Tests")
        # plt.plot(range(0,int(simulation_params['time_periods'])), re_change_order(a_tests_vec)[group], label="A Tests")
        plt.ylim(-max(float(K_mtest),float(K_atest))/10,max(float(K_mtest),float(K_atest))+max(float(K_mtest),float(K_atest))/10)
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*7)
        plt.plot(time_axis_controls, lockdowns_sim[group]["home"]+0.01, label="Home")
        plt.plot(time_axis_controls, lockdowns_sim[group]["work"]+0.01*2, label="Work")
        plt.plot(time_axis_controls, lockdowns_sim[group]["transport"]+0.01*3, label="Transport")
        plt.plot(time_axis_controls, lockdowns_sim[group]["school"]+0.01*4, label="School")
        plt.plot(time_axis_controls, lockdowns_sim[group]["leisure"]+0.01*5, label="Leisure")
        plt.plot(time_axis_controls, lockdowns_sim[group]["other"]+0.01*6, label="Other")
        plt.ylim(-0.1,1.1)
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*8)
        plt.plot(time_axis_controls, dynModel.groups[group].B_H, label="Bounced H")
        plt.plot(time_axis_controls, dynModel.groups[group].B_ICU, label="Bounced ICU")
        plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].B_H),np.max(dynModel.groups[group].B_ICU)) for group in groups]))
        plt.legend(loc='upper right')

    # Calulate number of contacts
    for i,group in enumerate(groups):
        plt.subplot(14,len(groups),i+1+len(groups)*9)
        for j,rec_group in enumerate(groups):
        	plt.plot(time_axis_controls, [dynModel.n_infections[t][group][rec_group]*death_prob[group] for t in range(T)], label=rec_group)
        plt.legend(loc='upper right')



    plt.subplot(14,2,21)
    #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
    plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
    #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
    plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
    plt.legend(loc='upper right')

    plt.subplot(14,2,22)
    #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
    plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
    #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
    plt.legend(loc='upper right')

    figure = plt.gcf() # get current figure
    figure.set_size_inches(7*len(groups),24)
    
    folder = "results/"+result['filename']+".pdf"
    print(folder)
    plt.savefig(folder)

    plt.close('all')





heuristics = ["real","full_open","full_lockdown","constant_gradient","time_gradient","age_group_gradient","linearization_heuristic", "linearization_heuristic_Prop_Bouncing"]
#heuristics = ["real","full_open","full_lockdown"]


all_data = []

for h in heuristics:
	for n in os.listdir("results/%s/"%h):
		if n[0:2] == "xi" and (".pdf" not in n):

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

			dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, 1, experiment_params["T"], universe_params["mixing"], start_day, result["experiment_params"]['eta'], extra_data = True)

			for t in range(experiment_params["T"]):
				dynModel.take_time_step(result["m_tests"][t], result["a_tests"][t], result["policy"][t])

			dynModel.take_end_steps()

			# Plot
			simulation_params = {
				'dt':1.0,
				'days': 90.0,
				'region': "fitted",
				'heuristic': result["lockdown_heuristic"],
				'mixing_method': {'name': 'multi'}
			}
			death_prob = death_prob_all if (result["groups"]=="all") else death_prob_one

			plot_benchmark(dynModel, result)

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
			else:
				data["lock_freq"] = result["experiment_params"]["policy_freq"],
			
			all_data.append(data)

pd.DataFrame(all_data).to_excel("results/results.xlsx")

