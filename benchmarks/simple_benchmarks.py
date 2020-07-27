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
    fig = plt.figure()
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
    plt.savefig("../results_runs/linearization_heuristic_dyn_models/"+simulation_params['region']+"_"+simulation_params['heuristic']+"_heuristic"+"_n_days_"+str(T)+"_tests_"+str(tests)+ "_icu_cap_"+str(dynModel.icus)+"_deltaS_"+str(delta)+"_xi_"+str(xi)+"_mixing_" + simulation_params['mixing_method']["name"]+"_benckmark_"+benchmark+"_testing"+testing+".pdf")




# Global variables
simulation_params = {
	'dt':1.0,
	'days': 90.0,
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
	"xi":[0, 30 * 37199.03],
	"icus":[2000],
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
					pickle.dump(dynModel,open(f"dynModel_real_benchmark_days_{simulation_params['time_periods']}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}.p","wb"))

					plot_benchmark(dynModel, delta, xi, icus, tests, testing, simulation_params, "real")


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
					pickle.dump(dynModel,open(f"dynModel_gov_full_lockd_benchmark_days_{simulation_params['time_periods']}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}.p","wb"))

					plot_benchmark(dynModel, delta, xi, icus, tests, testing, simulation_params, "govm_full_lockdown")


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
					pickle.dump(dynModel,open(f"dynModel_zero_full_lock_benchmark_days_{simulation_params['time_periods']}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}.p","wb"))

					plot_benchmark(dynModel, delta, xi, icus, tests, testing, simulation_params, "zero_full_lockdown")

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
					pickle.dump(dynModel,open(f"dynModel_open_benchmark_days_{simulation_params['time_periods']}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}.p","wb"))
					plot_benchmark(dynModel, delta, xi, icus, tests, testing, simulation_params, "open")

pd.DataFrame(results).to_excel(f"simulations-{simulation_params['days']}-days.xlsx")
