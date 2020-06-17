#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:11:42 2020

@author: dragosciocan
"""

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

from group import SEIR_group, DynamicalModel
from heuristics import *
import linearization
# from forecasting_heuristic import *
import math
import pprint
import time

def run_linearization_heuristic(simulation_params):

    start_time = time.time()

    # Define time variables
    num_time_periods = int(math.ceil(simulation_params["num_days"]/simulation_params["dt"]))

    # Define mixing method
    mixing_method = simulation_params['mixing_method']

    # Read group parameters
    with open("parameters/"+simulation_params["region"]+".yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        universe_params = yaml.load(file, Loader=yaml.FullLoader)

        # Read initialization
    with open("initialization/initialization.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        initialization = yaml.load(file, Loader=yaml.FullLoader)

    # Define policy
    with open('benchmarks/static_infected_10.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        policy_file = yaml.load(file, Loader=yaml.FullLoader)
    alphas_vec = policy_file['alphas_vec']

    # Percentage infected at time 0
    perc_infected = simulation_params['perc_infected']
    # Move population to infected (without this there is no epidem.)
    for group in initialization:
    	change = initialization[group]["S"]*perc_infected/100
    	initialization[group]["S"] = initialization[group]["S"] - change
    	initialization[group]["I"] = initialization[group]["I"] + change

    dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], num_time_periods, mixing_method)

    # add parameters for testing capacity
    dynModel.parameters['global-parameters']['C_mtest'] = simulation_params['mtest_cap']
    dynModel.parameters['global-parameters']['C_atest'] = simulation_params['atest_cap']

    linearization.run_heuristic_linearization(dynModel)


    end_time = time.time()

    print("Total running time for {} days is {}".format(simulation_params['num_days'], end_time - start_time))

    return dynModel

def run_nl_l_heuristic(simulation_params):
    n_days = simulation_params['num_daysToLockDown'] + simulation_params['num_daysAfterLockDown']

    # Define time variables
    num_time_periods = int(math.ceil(n_days/simulation_params["dt"]))

    # Define mixing method
    mixing_method = simulation_params['mixing_method']

    # Read group parameters
    with open("parameters/"+simulation_params["region"]+".yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        universe_params = yaml.load(file, Loader=yaml.FullLoader)

    # Read initialization
    with open("initialization/patient_zero.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        initialization = yaml.load(file, Loader=yaml.FullLoader)

    # Update initialization
    # Put exactly initial_infected infected individuals in age group 40-49. No infected individuals in other groups.
    initialization["age_group_40_49"]["I"] = initialization["age_group_40_49"]["I"] + int(simulation_params['initial_infected_count'])
    initialization["age_group_40_49"]["S"] = initialization["age_group_40_49"]["S"] - int(simulation_params['initial_infected_count'])

    # Read lockdown
    activities_non_home = ['leisure', 'other', 'school', 'transport', 'work']
    actions_dict = {}
    for group in initialization:
        actions_dict[group] = {}
        actions_dict[group][0] = {}
        actions_dict[group][1] = {}
        actions_dict[group][0]['home'] = 1
        actions_dict[group][1]['home'] = 1
        for act in activities_non_home:
            actions_dict[group][0][act] = float(simulation_params['work_full_lockdown_factor'])
            actions_dict[group][1][act] = 1

    # Define policy
    total_lockdown_pattern = "000000000"
    total_lockdown = {
    	'age_group_%d_%d'%(10*i,10*i+9):actions_dict['age_group_%d_%d'%(10*i,10*i+9)][int(total_lockdown_pattern[i])] for i in range(0,8)
    }
    total_lockdown['age_group_80_plus'] = actions_dict['age_group_80_plus'][0]

    no_lockdown_pattern = "111111111"
    no_lockdown = {
    	'age_group_%d_%d'%(10*i,10*i+9):actions_dict['age_group_%d_%d'%(10*i,10*i+9)][int(no_lockdown_pattern[i])] for i in range(0,8)
    }
    no_lockdown['age_group_80_plus'] = actions_dict['age_group_80_plus'][1]

    # Create environment
    dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], num_time_periods, mixing_method)

    # add parameters for testing capacity
    dynModel.parameters['global-parameters']['C_mtest'] = simulation_params['mtest_cap']
    dynModel.parameters['global-parameters']['C_atest'] = simulation_params['atest_cap']

    # Construct vector of tests with a heuristic
    max_m_tests = [dynModel.parameters['global-parameters']['C_mtest'] for t in range(num_time_periods)]
    max_a_tests = [dynModel.parameters['global-parameters']['C_atest'] for t in range(num_time_periods)]
    if (simulation_params['testing_policy'] == "random"):
        groups = []
        for group in dynModel.parameters['seir-groups']:
            population = sum([dynModel.initialization[group][sg] for sg in ["S","E","I","R","Ia","Ips","Ims","Iss","Rq","H","ICU","D"]])
            if population > 0:
                groups.append(group)
        groups.sort()
        a_tests_vec, m_tests_vec = random_partition(dynModel, groups, max_a_tests, max_m_tests)

    elif (simulation_params['testing_policy'] == "homogeneous"):
        a_tests_vec, m_tests_vec = homogeneous(dynModel, max_a_tests, max_m_tests)
    elif (simulation_params['testing_policy'] == "no_tests"):
        a_tests_vec, m_tests_vec = no_tests(dynModel)

    #ICU CAP replaced by single value dynModel.icus
    tests = {
    	'a_tests_vec':a_tests_vec,
    	'm_tests_vec':m_tests_vec,
    }

    for t in range(0,int(simulation_params['num_daysToLockDown'])):
        dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], no_lockdown)
    for t in range(int(simulation_params['num_daysToLockDown']),int(simulation_params['num_daysToLockDown'] + simulation_params['num_daysAfterLockDown'])):
        dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], total_lockdown)

    return dynModel



def main():

    # simulation_params_l_nl_heuristic = {
    #     'dt':1.0,
    #     'region': "Ile-de-France",
    #     'quar_freq': 1,
    #     'num_daysToLockDown' : 10,
    #     'num_daysAfterLockDown' : 10,
    #     'initial_infected_count' : 1,
    #     'perc_infected' : -1,
    #     'mixing_method' : {
    #         "name":"mult",
    #         "param_alpha":1.0,
    #         "param_beta":0.5,},
    #     'mtest_cap' : 100,
    #     'atest_cap' : 100,
    #     'testing_policy' : "homogeneous",
    #     'work_full_lockdown_factor' : 0.24,
    #     'heuristic': 'only_testing_homogenous'
    # }

    simulation_params_linearization = {
        'dt':1.0,
        'region': "Ile-de-France",
        'quar_freq': 1,
        'num_days' : 30,
        'initial_infected_count' : 1,
        'perc_infected' : 10,
        'mixing_method' : {
            "name":"mult",
            "param_alpha":1.0,
            "param_beta":0.5,},
        'mtest_cap' : 100,
        'atest_cap' : 100,
        'work_full_lockdown_factor' : 0.24,
        'heuristic': 'linearization'
    }

    # run_nl_l_heuristic(simulation_params_l_nl_heuristic)
    dynModel_linearization_heur = run_linearization_heuristic(simulation_params_linearization)

    plot_and_print_results(dynModel_linearization_heur, simulation_params_linearization)




def plot_and_print_results(dynModel, simulation_params):
    ##############################################################################
    # PLOT AND PRINT RESULTS FROM THE DYNAMICAL MODEL SIMULATED USING THE SIMULATION PARAMS

    heuristic = simulation_params['heuristic']
    T = dynModel.time_steps

    K_mtest = simulation_params['mtest_cap']
    K_atest = simulation_params['atest_cap']

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
    plt.figure(1)
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
    figure.suptitle('Region: %s, %s Heuristic with Total Days: %s, Initial Infected percentage: %2d, M-test daily capacity: %s, A-test daily capacity: %s, Mixing: %s'%(simulation_params['region'],simulation_params['heuristic'],T,simulation_params['perc_infected'],K_mtest,K_atest,simulation_params['mixing_method']["name"]), fontsize=22)
    plt.savefig("results_runs/optimization/"+simulation_params['region']+"_"+simulation_params['heuristic']+"_heuristic"+"_n_days_"+str(T)+"_initial_infected_percentage_"+str(simulation_params['perc_infected'])+"_m_tests_"+str(dynModel.parameters['global-parameters']['C_mtest'])+"_a_tests_"+str(dynModel.parameters['global-parameters']['C_atest'])+"_mixing_" + simulation_params['mixing_method']["name"]+".pdf")




if __name__ == "__main__":
    main()
