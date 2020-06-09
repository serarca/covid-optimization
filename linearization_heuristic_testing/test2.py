#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 21:32:08 2020

@author: spyroszoumpoulis1
"""


import os.path
import sys
from inspect import getsourcefile
from random import *

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/heuristics")

from linearization import *
from heuristics import *

# Global variables
simulation_params = {
        'dt':1.0,
        'days': 3,
        'region': "Ile-de-France",
        'quar_freq': 3,
}

# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

# Define mixing method
mixing_method = {
    "name":"mult",
    "param_alpha":1.0,
    "param_beta":0.5,
    #"param":float(args.mixing_param) if args.mixing_param else 0.0,
}

# Read group parameters
with open("../parameters/"+simulation_params["region"]+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("../initialization/initialization.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    initialization = yaml.load(file, Loader=yaml.FullLoader)

# Define policy
with open('../benchmarks/static_infected_10.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    policy_file = yaml.load(file, Loader=yaml.FullLoader)
alphas_vec = policy_file['alphas_vec']

# Percentage infected at time 0
perc_infected = 10
# Move population to infected (without this there is no epidem.)
for group in initialization:
	change = initialization[group]["S"]*perc_infected/100
	initialization[group]["S"] = initialization[group]["S"] - change
	initialization[group]["I"] = initialization[group]["I"] + change

# Create environment
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
#print(dynModel.time_steps)

# Set up testing decisions: no testing for now
a_tests_vec, m_tests_vec = no_tests(dynModel)
tests = {
    'a_tests_vec':a_tests_vec,
    'm_tests_vec':m_tests_vec,
}


# ############# Testing get_X_hat_sequence #####################
#
# # Obtain a sequence of tests to compare the X_hat seq with
# # the dynamics of a "parallel", identical dynamical model
daily_tests = 1e4
max_a_tests = [daily_tests for i in range(dynModel.time_steps)]
#print(len(max_a_tests))
max_m_tests = [daily_tests for i in range(dynModel.time_steps)]
a_tests_vec, m_tests_vec = homogeneous(dynModel, max_a_tests, max_m_tests)

# Create an identical, "parallel" dynamical model. Simulate it to the end of the horizon
dynModel2 = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
dynModel2.simulate(m_tests_vec, a_tests_vec, alphas_vec)


for k in range(0,dynModel.time_steps):
    print("k:",k)
    
    # Construct u_hat sequence, of dimensions (num_age_groups * num_controls, T-k)
    u_hat_sequence = np.zeros((num_age_groups * num_controls, dynModel.time_steps-k))
    
    for i in range(dynModel.time_steps - k):
        print("First i loop, i:",i)
        print("k+i:",k+i)
        for ag in range(num_age_groups):
            u_hat_sequence[ag * num_controls + controls.index('Nmtest_g'),i] = m_tests_vec[k+i][age_groups[ag]]
            #print(m_tests_vec[k+i][age_groups[ag]])

            u_hat_sequence[ag * num_controls + controls.index('Natest_g'),i] = a_tests_vec[k+i][age_groups[ag]]

            for act in activities:
                u_hat_sequence[ag * num_controls + controls.index(act),i] = alphas_vec[k+i][age_groups[ag]][act]

    print("Clock of dynModel:",dynModel.t)
    
    #Get X_hat_sequence
    X_hat_sequence = get_X_hat_sequence(dynModel, k, u_hat_sequence)
    #print(X_hat_sequence)
    
    print("Clock of dynModel:",dynModel.t)

    #Check that the states of the dyn model are the same as X_hat
    for i in range(dynModel.time_steps - k):
        print("Second i loop, i:",i)
        print("k+i:",k+i)
        for ag in range(num_age_groups):
            for st in SEIR_groups:
                # The assertion checks that for each compartment,
                # at each time and for each group, the values in
                # X_hat_sequence coincide with the values in the
                # states of dynModel2. st is the state/
                # compartment, and we get the names used un group
                # by removing the _g.

                # if (st=="S_g"):
                #     print(dynModel2.get_state(k+i)[age_groups[0]]['S_g'.replace('_g','')])
                
                assert(
                X_hat_sequence[ag * num_compartments + SEIR_groups.index(st),i]
                ==
                dynModel2.get_state(k+i)[age_groups[ag]][st.replace('_g','')])
                
    if (k<dynModel.time_steps):
        m_tests = {}
        a_tests = {}
        for g in age_groups:
            m_tests[g] = daily_tests/(num_age_groups+0.0)
            a_tests[g] = daily_tests/(num_age_groups+0.0)
    
        u_hat_dict, alphas = buildAlphaDict(u_hat_sequence[:,0])
        dynModel.take_time_step(m_tests, a_tests, alphas)
        print("Clock of dynModel:",dynModel.t)

print("Finished succesfully the equality test for X_hat_sequence. It coincides with the states in dynModel2 every time.")