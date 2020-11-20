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


def plot_benchmark(dynModel, delta, xi, icus, tests, testing, simulation_params, benchmark):

    T = dynModel.time_steps

    K_mtest = tests
    K_atest = tests

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
    time_axis = [i*simulation_params["dt"] for i in range(T+1)]
    time_axis_controls = [i*simulation_params["dt"] for i in range(T)]

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

    plt.subplot(13,2,19)
    #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
    plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
    #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
    plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
    plt.legend(loc='upper right')

    plt.subplot(13,2,20)
    #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
    plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
    #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
    plt.legend(loc='upper right')

    figure = plt.gcf() # get current figure
    figure.set_size_inches(7*len(groups),24)
    figure.suptitle('Region: %s, %s Heuristic with Total Days: %s, M-test daily capacity: %s, A-test daily capacity: %s, '%(simulation_params['region'],simulation_params['heuristic'],T,K_mtest,K_atest), fontsize=22)
    plt.savefig("./"+benchmark+"/"+simulation_params['region']+"_"+simulation_params['heuristic']+"_heuristic"+"_n_days_"+str(T)+"_tests_"+str(tests)+ "_icu_cap_"+str(dynModel.icus)+"_deltaS_"+str(delta)+"_xi_"+str(xi)+"_mixing_" + simulation_params['mixing_method']["name"]+"_benckmark_"+benchmark+"_testing"+testing+".pdf")

    plt.close('all')



# Global variables
simulation_params = {
	'dt':1.0,
	'days': 90.0,
	'region': "fitted",
	'heuristic': 'benchmark',
	'mixing_method': {'name': 'multi'}
}
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']


cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]
activities = ['home','leisure','other','school','transport','work']
rel_activities = ['leisure','other','school','transport','work']


# age_groups = ['age_group_30_39']

# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))


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

# Read econ parameters
with open("../lower_bounds/fitted.yaml") as file:
	lower_bounds = yaml.load(file, Loader=yaml.FullLoader)

eta=0.1

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



# Parameters to try
params_to_try = {
	"delta_schooling":[0.5],
	"xi":[0,1e6],
	"icus":[3000],
	"tests":[0],
	"testing":["homogeneous"]
}

thresholds_to_try = {
	"icu_utilization_rate":[0.1*(i+1) for i in range(10)], # Revisit this
	"incidence_rate":[0.0001*(i+1) for i in range(40)], # Revisit this
	"incidence_rate_old":[0.0001*(i+1) for i in range(40)], # Revisit this
}

lockdowns_to_try = {
	"low_activity":[0.05*i for i in range(21)],
	"high_activity":[0.05*i for i in range(21)]
}


def French_trigger_policy(experiment_params, thresholds, activity_levels, plot=False):
	
	# Define the high and low activity levels
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
	
	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day, eta)
	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}
	
	# Initialize alpha with high level for all groups and activities
	alpha = {ag:high_activity_policy for ag in age_groups}
	sevendays_newInf = []
	sevendays_newInf_old = []
	
	for t in range(simulation_params['time_periods']):
		# Recover the state before taking a time step, so as to get the "E" state before taking the next step. 
		# This will yield the new infections in the next step.
		state = dynModel.get_state(t)
		if (t==0):
			initial_population = np.sum([state[ag]["N"] for ag in age_groups])
			initial_population_old = np.sum([state[ag]["N"] for ag in ['age_group_60_69', 'age_group_70_79', 'age_group_80_plus']])
		
		# Find incidence rate over last 7 days in overall population
		newInfections = np.sum([dynModel.groups[ag].parameters['sigma']*state[ag]["E"] for ag in age_groups])
		sevendays_newInf.append(newInfections)
		# Drop first entry if larger than 7
		if len(sevendays_newInf)>7:
			sevendays_newInf = sevendays_newInf[1:8]
		incidence_rate = np.sum(sevendays_newInf)/initial_population
		
		# Find incidence rate over last 7 days in population aged 60+
		newInfections_old = np.sum([dynModel.groups[ag].parameters['sigma']*state[ag]["E"] for ag in ['age_group_60_69', 'age_group_70_79', 'age_group_80_plus']])
		sevendays_newInf_old.append(newInfections_old)
		# Drop first entry if larger than 7
		if len(sevendays_newInf_old)>7:
			sevendays_newInf_old = sevendays_newInf_old[1:8]
		incidence_rate_old = np.sum(sevendays_newInf_old)/initial_population_old
		
		# Take a time step. The ICU utilization rate will be calculated using the resulting ICU state.
		result = dynModel.take_time_step(m_tests, a_tests, alpha)
		new_state = result["state"]
		
		# Find ICU utilization rate
		icu_utilization_rate = np.sum([new_state[ag]["ICU"] for ag in age_groups])/experiment_params["icus"]
		
		
		if icu_utilization_rate>thresholds["icu_utilization_rate"] and incidence_rate>thresholds["incidence_rate"] and incidence_rate_old>thresholds["incidence_rate_old"]:
			alpha = {ag:low_activity_policy for ag in age_groups}
		else:
			alpha = {ag:high_activity_policy for ag in age_groups}
	
	
	result = {
		"heuristic":"French_trigger",
		"delta_schooling":experiment_params["delta_schooling"],
		"xi":experiment_params["xi"],
		"icus":experiment_params["icus"],
		"tests":experiment_params["tests"],
		"testing":experiment_params["testing"],
		"economics_value":dynModel.get_total_economic_value(),
		"deaths":dynModel.get_total_deaths(),
		"reward":dynModel.get_total_reward(),	
	}

	if plot:
		plot_benchmark(dynModel, 
			experiment_params["delta_schooling"], 
			experiment_params["xi"], 
			experiment_params["icus"], 
			experiment_params["tests"], 
			experiment_params["testing"], 
			simulation_params, 
			"French_trigger")

	return result


# Finding best parameter settings for French trigger policy
partial_results = []
all_results = []
for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					experiment_params = {
						'delta_schooling':delta,
						'xi':xi,
						'icus':icus,
						'testing':testing,
						'tests':tests,
					}

					best_reward = -float("inf")
					best_result = 0
					for icu_util_rate_t in thresholds_to_try["icu_utilization_rate"]:
						for inc_rate_t in thresholds_to_try["incidence_rate"]:
							for inc_rate_old_t in thresholds_to_try["incidence_rate_old"]:
								for act_H_t in lockdowns_to_try["high_activity"]:
									for act_L_t in lockdowns_to_try["low_activity"]:
										

										thresholds = {
											#"icus_t":icus_t,
											#"beds_t":beds_t,
											"icu_utilization_rate":icu_util_rate_t,
											"incidence_rate":inc_rate_t,
											"incidence_rate_old":inc_rate_old_t,
										}
										activity_levels = {
											"high_activity":act_H_t,
											"low_activity":act_L_t,
										}
								
										partial_result = French_trigger_policy(experiment_params, thresholds, activity_levels)
										partial_result.update(thresholds)
										partial_result.update(activity_levels)
										partial_results.append(partial_result)
										if partial_result["reward"]>best_reward:
											best_reward = partial_result["reward"]
											best_result = partial_result
											print(best_result)
										#full_open_policy["school"] = l_school

					all_results.append(best_result)


					#pickle.dump(dynModel,open(f"dynModel_gov_full_lockd_benchmark_days_{simulation_params['time_periods']}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}.p","wb"))

# We plot the winning model
for r in all_results:
	experiment_params = {
		"delta_schooling":r['delta_schooling'],
		"xi":r['xi'],
		"icus":r['icus'],
		"tests":r['tests'],
		"testing":r['testing'],
	}
	thresholds = {
		"icu_utilization_rate":r["icu_utilization_rate"],
		"incidence_rate":r["incidence_rate"],
		"incidence_rate_old":r["incidence_rate_old"],
	}	

	French_trigger_policy(experiment_params, thresholds, activity_levels, plot=True)


pd.DataFrame(all_results).to_excel(f"French_trigger_simulations-{simulation_params['days']}-days.xlsx")
pd.DataFrame(partial_results).to_excel(f"French_trigger_partial_simulations-{simulation_params['days']}-days.xlsx")

