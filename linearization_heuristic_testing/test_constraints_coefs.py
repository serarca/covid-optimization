#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 16:34:25 2020

@author: spyroszoumpoulis1
"""

import os.path
import sys
from inspect import getsourcefile
from random import *
import numpy as np
import numpy
import pandas as pd
import math
import gurobipy as gb

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
        'days': 50,
        'region': "Ile-de-France",
        'quar_freq': 182,
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

# Set up testing decisions: no testing for now
a_tests_vec, m_tests_vec = no_tests(dynModel)
tests = {
    'a_tests_vec':a_tests_vec,
    'm_tests_vec':m_tests_vec,
}

dynModel.simulate(m_tests_vec, a_tests_vec, alphas_vec)


# ############# Testing calculate_generic_constraint_coefs #####################

# # Obtain a sequence of tests
daily_tests = 1e4
max_a_tests = [daily_tests for i in range(dynModel.time_steps + 1)]
#print(len(max_a_tests))
max_m_tests = [daily_tests for i in range(dynModel.time_steps + 1)]
a_tests_vec, m_tests_vec = homogeneous(dynModel, max_a_tests, max_m_tests)

k=27

# Construct u_hat sequence, of dimensions (num_age_groups * num_controls, T-k+1)
u_hat_sequence = np.zeros((num_age_groups * num_controls, dynModel.time_steps-k+1))
for i in range(dynModel.time_steps - k + 1):
    #print("First i loop, i:",i)
    #print("k+i:",k+i)
    for ag in range(num_age_groups):
        u_hat_sequence[ag * num_controls + controls.index('Nmtest_g'),i] = m_tests_vec[k+i][age_groups[ag]]
        #print(m_tests_vec[k+i][age_groups[ag]])

        u_hat_sequence[ag * num_controls + controls.index('Natest_g'),i] = a_tests_vec[k+i][age_groups[ag]]

        for act in activities:
            u_hat_sequence[ag * num_controls + controls.index(act),i] = alphas_vec[k+i][age_groups[ag]][act]




#
# a_ICU, b_ICU = calculate_ICU_constraint_coefs(dynModel)
# a_H, b_H = calculate_H_constraint_coefs(dynModel)
# a_BH_0_9, b_BH_0_9 = calculate_BH_constraint_coefs(dynModel,'age_group_0_9')
# a_BICU_0_9, b_BICU_0_9 = calculate_BICU_constraint_coefs(dynModel,'age_group_0_9')
# a_BH_80_plus, b_BH_80_plus = calculate_BH_constraint_coefs(dynModel,'age_group_80_plus')
# a_BICU_80_plus, b_BICU_80_plus = calculate_BICU_constraint_coefs(dynModel,'age_group_80_plus')
# #a_matrix = numpy.column_stack((a_H,a_ICU,a_BH_0_9,a_BICU_0_9,a_BH_80_plus,a_BICU_80_plus))
# #b_matrix = numpy.column_stack((b_H,b_ICU,b_BH_0_9,b_BICU_0_9,b_BH_80_plus,b_BICU_80_plus))
# a_matrix = np.row_stack((a_H,a_ICU,a_BH_0_9,a_BICU_0_9,a_BH_80_plus,a_BICU_80_plus))
# b_matrix = np.row_stack((b_H,b_ICU,b_BH_0_9,b_BICU_0_9,b_BH_80_plus,b_BICU_80_plus))
#
# get the X_hat_sequence
X_hat_sequence = get_X_hat_sequence(dynModel, k, u_hat_sequence)
# Xhat_seq = np.random.rand(num_age_groups * num_compartments, dynModel.time_steps-k+1)
# u_hat_sequence = np.random.rand(num_age_groups * num_controls, dynModel.time_steps-k+1)
#
# # u_coeffs, constants = calculate_all_constraint_coefs(dynModel, k, Xhat_seq, u_hat_sequence, a_matrix, b_matrix)
# # Print u_coeffs for BICU constraint for age_group_80_plus
# #
# print("BICU constraint for age_group_80_plus, t=k, u_k:\n",u_coeffs[k][5][0,80:90]) #this should output 1 for Bounce ICU
# print("BICU constraint for age_group_80_plus, t=k, u_{k+10}:\n",u_coeffs[k][5][10,80:90]) #this should output all zeros
# print("BICU constraint for age_group_80_plus, t=k+10, u_{k}:\n",u_coeffs[k+10][5][0,80:90]) #
# print("BICU constraint for age_group_80_plus, t=k+10, u_{k+5}:\n",u_coeffs[k+10][5][5,80:90]) #
# print("BICU constraint for age_group_80_plus, t=k+10, u_{k+10}:\n",u_coeffs[k+10][5][10,80:90]) #this should output 1 for Bounce ICU
# print("BICU constraint for age_group_80_plus, t=k+10, u_{k+11}:\n",u_coeffs[k+10][5][11,80:90]) #this should output all zeros
# print("BICU constraint for age_group_80_plus, t=k+10, u_{T}:\n",u_coeffs[k+10][5][dynModel.time_steps-k,80:90]) #this should output all zeros

Gamma_x, Gamma_u, K = calculate_all_constraints(dynModel)

d_matrix, e_matrix = calculate_objective_time_dependent_coefs(dynModel, k, X_hat_sequence, u_hat_sequence)

u_constr_coeffs, constr_constants, u_obj_coeffs = calculate_all_coefs(dynModel, k, Xhat_seq, uhat_seq, Gamma_x, Gamma_u, d_matrix, e_matrix)
