# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:08:59 2020

@author: Dan
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
        'days': 10,
        'region': "Ile-de-France",
        'quar_freq': 1,
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

# add parameters for testing capacity
dynModel.parameters['global-parameters']['C_mtest'] = 10000
dynModel.parameters['global-parameters']['C_atest'] = 10000

# ##############################################################################
# Testing the construction of a typical LP

# shorthand for a few useful parameters
T = dynModel.time_steps
Xt_dim = num_compartments * num_age_groups
ut_dim = num_controls * num_age_groups
num_constraints = 4 + 2*num_age_groups + num_age_groups*num_activities

# calculate M, gamma, eta
M, gamma, eta = calculate_M_gamma_and_eta(dynModel)
assert( np.shape(M) == (ut_dim,Xt_dim) )
assert( np.shape(gamma) == (Xt_dim,) )
assert( np.shape(eta) == (Xt_dim,) )

#########
# calculate all the constraints and store them
A, B, K = calculate_all_constraints(dynModel)

assert( np.shape(A) == (num_constraints,Xt_dim) )
assert( np.shape(B) == (num_constraints,ut_dim) )
assert( np.shape(K) == (num_constraints,T) )

# uptimal decisions
uopt_seq = np.zeros((ut_dim,T))

# pick a starting u_hat sequence; for now, no testing
uhat_seq = np.zeros((ut_dim,T))

for k in range(T):

    # calculate state trajectory X_hat
    Xhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq)
    assert( np.shape(Xhat_seq) == (Xt_dim,T-k) )

    # calculate objective parameters d, e
    D,E = calculate_objective_time_dependent_coefs(dynModel, k, Xhat_seq, uhat_seq)

    # get coefficients for decisions in all constraints and objective
    constr_coefs, constr_consts, obj_coefs = calculate_all_coefs(dynModel,k,Xhat_seq,uhat_seq,A,B,D,E)

    assert( np.shape(obj_coefs) == (ut_dim,T-k) )
    assert( len(constr_coefs) == T-k )
    assert( len(constr_consts) == T-k )
    for t in range(k,T):
        assert( len(constr_coefs[t]) == num_constraints )
        assert( len(constr_consts[t]) == num_constraints )

        for i in range(num_constraints):
            assert( np.shape(constr_coefs[t][i])==np.shape(uhat_seq) )
            assert( np.shape(constr_consts[t][i])==() )

    # create empty model
    mod = gb.Model("Linearization Heuristic")

    # add all decisions using matrix format, and also specify objective coefficients
    u_vars = mod.addMVar(np.shape(uhat_seq), obj=obj_coefs, name="u")
    
    ones_row = np.ones(ut_dim)
    ones_col = np.ones(T-k)
    for t in range(k,T):
        for con in range(num_constraints):
            mod.addConstr( ones_row @ (u_vars*constr_coefs[t][con]) @ ones_col + constr_consts[t][con] <= K[con,t] )

    # optimize the model
    mod.optimize()

