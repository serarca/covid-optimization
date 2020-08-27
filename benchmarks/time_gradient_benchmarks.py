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
with open("../initialization/50days.yaml") as file:
	initialization = yaml.load(file, Loader=yaml.FullLoader)
	start_day = 60

# Read econ parameters
with open("../parameters/econ.yaml") as file:
	econ_params = yaml.load(file, Loader=yaml.FullLoader)

# Read econ parameters
with open("../lower_bounds/fitted.yaml") as file:
	lower_bounds = yaml.load(file, Loader=yaml.FullLoader)


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
	"xi":[1e6],
	"icus":[3000],
	"tests":[0],
	"testing":["homogeneous"]
}

experiment_params = {
	"delta_schooling":0.5,
	"xi":0,
	"icus":3000,
	"tests":0,
	"testing":"homogeneous",
}


def run_government_policy(experiment_params):


	alphas_vec = []

	for t in range(simulation_params['time_periods']):
		index = t+start_day
		if t+start_day >= len(gov_policy):
			alphas_vec.append({ag:gov_policy[-1] for ag in age_groups})
		else:
			alphas_vec.append({ag:gov_policy[t+start_day] for ag in age_groups})

	# Create dynamical method
	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day)
	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}

	for t in range(simulation_params['time_periods']):
		dynModel.take_time_step(m_tests, a_tests, alphas_vec[t])

	result = {
		"heuristic":"real",
		"delta_schooling":experiment_params["delta_schooling"],
		"xi":experiment_params["xi"],
		"icus":experiment_params["icus"],
		"tests":experiment_params["tests"],
		"testing":experiment_params["testing"],
		"economics_value":dynModel.get_total_economic_value(),
		"deaths":dynModel.get_total_deaths(),
		"reward":dynModel.get_total_reward(),	
	}

	return result

def run_time_policy(experiment_params, alphas, plot=False):
	# Create dynamical method
	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day)
	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}

	for t in range(simulation_params['time_periods']):
		dynModel.take_time_step(m_tests, a_tests, alphas[t])

	if plot:
			plot_benchmark(dynModel, 
			experiment_params["delta_schooling"], 
			experiment_params["xi"], 
			experiment_params["icus"], 
			experiment_params["tests"], 
			experiment_params["testing"], 
			simulation_params, 
			"time_gradient")

	result = {
		"heuristic":"time_gradient",
		"delta_schooling":experiment_params["delta_schooling"],
		"xi":experiment_params["xi"],
		"icus":experiment_params["icus"],
		"tests":experiment_params["tests"],
		"testing":experiment_params["testing"],
		"economics_value":dynModel.get_total_economic_value(),
		"deaths":dynModel.get_total_deaths(),
		"reward":dynModel.get_total_reward(),	
	}

	return result


def run_constant_policy(experiment_params, alpha, plot=False):

	# Create dynamical method
	dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day)
	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}

	for t in range(simulation_params['time_periods']):
		dynModel.take_time_step(m_tests, a_tests, alpha)

	if plot:
			plot_benchmark(dynModel, 
			experiment_params["delta_schooling"], 
			experiment_params["xi"], 
			experiment_params["icus"], 
			experiment_params["tests"], 
			experiment_params["testing"], 
			simulation_params, 
			"time_gradient")

	result = {
		"heuristic":"time_gradient",
		"delta_schooling":experiment_params["delta_schooling"],
		"xi":experiment_params["xi"],
		"icus":experiment_params["icus"],
		"tests":experiment_params["tests"],
		"testing":experiment_params["testing"],
		"economics_value":dynModel.get_total_economic_value(),
		"deaths":dynModel.get_total_deaths(),
		"reward":dynModel.get_total_reward(),	
	}

	return result

def run_full_lockdown(experiment_params):

	ag_alpha = gov_policy[start_lockdown]

	alpha = {
		ag:ag_alpha for ag in age_groups
	}

	result = run_constant_policy(experiment_params, alpha)
	result["heuristic"] = "full-lockdown"
	return result

def run_open(experiment_params):
	ag_alpha = {
		"home": 1.0,
		"leisure": 1.0,
		"other": 1.0,
		"school": 1.0,
		"transport": 1.0,
		"work": 1.0
	}

	alpha = {
		ag:ag_alpha for ag in age_groups
	}

	result = run_constant_policy(experiment_params, alpha)
	result["heuristic"] = "full-open"
	return result



def gradient_descent(experiment_params, quar_freq, plot = False):

	intervention_times = [t*quar_freq for t in range(int(simulation_params['days']/quar_freq))]

	# Read the constant alphas to initialize the solution
	with open('./constant_gradient_alphas/delta_%f_xi_%d_icus_%d_tests_%d_testing_%s.yaml'%(experiment_params["delta_schooling"],experiment_params["xi"],experiment_params["icus"],experiment_params["tests"], experiment_params["testing"])) as file:
		initial_alphas = yaml.load(file, Loader=yaml.FullLoader)


	x0 = np.zeros(len(intervention_times)*len(rel_activities))
	for i,t in enumerate(intervention_times):
		for k,act in enumerate(rel_activities):
			x0[i*len(rel_activities) + k] = initial_alphas["age_group_30_39"][act]



	# Create dynamical model
	fastModel = FastDynamicalModel(universe_params, econ_params, experiment_params, simulation_params['dt'], mixing_method)
	initial_state = state_to_matrix(initialization)

	if experiment_params["testing"] == "homogeneous":
		m_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}
		a_tests = {ag:experiment_params["tests"]/len(age_groups) for ag in age_groups}

	m_tests_vec = dict_to_vector(m_tests)
	a_tests_vec = dict_to_vector(a_tests)


	def simulate(x):
		# Extract vector components
		x_lockdown = np.zeros((len(intervention_times),len(age_groups),len(rel_activities)))


		x_aux = np.reshape(
			x,
			(len(intervention_times),len(rel_activities))
		)

		for i,t in enumerate(intervention_times):
			for j,ag in enumerate(age_groups):
				x_lockdown[i,j,:] = x_aux[i,:]



		state = initial_state
		total_reward = 0
		total_deaths = 0
		total_econ = 0
		for t in range(simulation_params['time_periods']):
			if t in intervention_times:
				update_contacts = True
			else:
				update_contacts = False

			if start_day + t < universe_params['days_before_gamma']:
				lockdown_status = "pre-gamma"
			else:
				lockdown_status = "post-gamma"
			
			# Add home activity to lockdown
			x_lockdown_all = np.concatenate((np.array([[1.0]]*len(age_groups)),x_lockdown[int(t/quar_freq),:,:]), axis=1)


			state, econs = fastModel.take_time_step(
				state, 
				m_tests_vec, 
				a_tests_vec,
				x_lockdown_all,
				lockdown_status,
				update_contacts = update_contacts, 
				B_H = False, 
				B_ICU = False,
				B_H_perc = False,
				B_ICU_perc = False,
			)
			total_reward += econs['reward']
			total_deaths += econs['deaths']
			total_econ += econs['economic_value']

		print(total_reward, total_deaths, total_econ)
		return -total_reward

	lower_bounds_matrix = np.zeros((len(intervention_times),len(rel_activities)))
	for i,act in enumerate(rel_activities):
		lower_bounds_matrix[:,i] += lower_bounds[act]


	
	full_bounds = Bounds(
		lower_bounds_matrix.flatten(),
		np.zeros(len(intervention_times)*len(rel_activities)) + 1.0
	)

	result_lockdown = minimize(simulate, x0, method='L-BFGS-B',bounds=full_bounds,options={'eps':10e-8,'maxfun':700000})


	x_aux = np.reshape(
			result_lockdown.x,
			(len(intervention_times),len(rel_activities))
		)


	x_lockdown = np.zeros((len(intervention_times),len(age_groups),len(rel_activities)))

	for i,t in enumerate(intervention_times):
		for j,ag in enumerate(age_groups):
			x_lockdown[i,j,:] = x_aux[i,:]


	alphas = matrix_to_alphas(x_lockdown, quar_freq)

	with open('./time_gradient_alphas/delta_%f_xi_%d_icus_%d_tests_%d_testing_%s.yaml'%(experiment_params["delta_schooling"],experiment_params["xi"],experiment_params["icus"],experiment_params["tests"], experiment_params["testing"]), 'w') as file:
		yaml.dump(alphas, file)

	result_policy = run_time_policy(experiment_params, alphas, plot)
	result_policy["heuristic"] = "time_gradient"

	return result_policy




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
					print(experiment_params)
					result_gradient = gradient_descent(experiment_params, 10, plot=True)
					all_results.append(result_gradient)





pd.DataFrame(all_results).to_excel(f"gradient_time_simulations-{simulation_params['days']}-days.xlsx")
