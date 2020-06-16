# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:08:59 2020

@author: Dan
"""

import os.path
import sys
import numpy as np
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
        'days': 20,
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

# some boolean flags for running the heuristic
use_bounce_var = True   # whether to use the optimal bounce variables when forecasting the new X_hat
bounce_existing = False   # whether to allow bouncing existing patients

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
Gamma_x, Gamma_u, K, all_labels = calculate_all_constraints(dynModel,bounce_existing)

assert( np.shape(Gamma_x) == (num_constraints,Xt_dim) )
assert( np.shape(Gamma_u) == (num_constraints,ut_dim) )
assert( np.shape(K) == (num_constraints,T) )

# uptimal decisions
uopt_seq = np.zeros((ut_dim,T))

# pick a starting u_hat sequence
uhat_seq = np.zeros((ut_dim,T))
# for now, homogenous testing
Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
uhat_seq[Nmtestg_idx_all,:] = dynModel.parameters['global-parameters']['C_mtest']/num_age_groups

Natestg_idx_all = slice(controls.index('Natest_g'),ut_dim,num_controls)
uhat_seq[Natestg_idx_all,:] = dynModel.parameters['global-parameters']['C_atest']/num_age_groups

# and home lockdown variables all 1
lock_home_idx_all = slice(controls.index('home'),ut_dim,num_controls)
uhat_seq[lock_home_idx_all,:] = 1.0

for k in range(T):

    print("\n\n TIME k= {}.".format(k))


    # calculate state trajectory X_hat and corresponging controls new_uhat
    Xhat_seq, new_uhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq, use_bounce_var)

    print("Finished getting nominal trajectory for time {}".format(k))
    print("-----------------------")

    assert( np.shape(Xhat_seq) == (Xt_dim,T-k) )
    assert( np.shape(new_uhat_seq) == (ut_dim,T-k) )

    # overwrite uhat with the updated one (with new bounce variables)
    #print("\nOld uhat at 1:")
    #print(uhat_seq[:,1])
    #print("\nNew uhat at 1")
    #print(new_uhat_seq[:,1])

    uhat_seq = new_uhat_seq

    ICUidx_all = slice(SEIR_groups.index('ICU_g'),Xt_dim,num_compartments)
    print("Total people in ICU at start of period k: {}".format(np.sum(Xhat_seq[ICUidx_all,0])))

    # calculate objective parameters d, e
    D,E = calculate_objective_time_dependent_coefs(dynModel, k, Xhat_seq, uhat_seq)

    print("Calculated obj. time dep coeff for time {}".format(k))
    print("-----------------------")

    # get coefficients for decisions in all constraints and objective
    constr_coefs, constr_consts, obj_coefs = calculate_all_coefs(dynModel,k,Xhat_seq,uhat_seq,Gamma_x,Gamma_u,D,E)




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
    mod.setParam( 'OutputFlag', False )     # make Gurobi silent
    mod.Params.DualReductions = 0  # change this to get explicit infeasible or unbounded

    # add all decisions using matrix format, and also specify objective coefficients
    # u_vars = mod.addMVar(np.shape(uhat_seq), obj=obj_coefs, name="u")
    obj_vec = np.reshape(obj_coefs, (ut_dim*(T-k),), 'F')  # reshape by reading along rows first
    u_vars_vec = mod.addMVar( np.shape(obj_vec), obj=obj_vec, name="u")

    mod.ModelSense = -1

    x_feas = np.zeros( (len(obj_vec),) )  # a feasible solution

    for t in range(k,T):
        #print("Time %d number of constraints %d" %(t,len(constr_coefs[t])))
        for con in range(num_constraints):
            cons_vec = np.reshape(constr_coefs[t][con], (len(obj_vec),), 'F')
            cname = ("%s[t=%d]" %(all_labels[con],t))
            mod.addConstr( u_vars_vec @ cons_vec + constr_consts[t][con] <= K[con,t], name=cname)

    mod.write("LP_lineariz_model.lp")       # write the LP to a file

    # optimize the model
    mod.optimize()

    if( mod.Status ==  gb.GRB.INFEASIBLE ):
        # model was infeasible
        mod.computeIIS()  # irreducible system of infeasible inequalities
        mod.write("LP_lineariz_IIS.ilp")
        print("ERROR. Problem infeasible at time k={}. Halting...".format(k))
        assert(False)

    # extract decisions for current period (testing and alphas)
    uvars_opt = np.reshape(u_vars_vec.X, np.shape(obj_coefs), 'F')
    uopt_seq[:,k] = uvars_opt[:,0]
    uk_opt_dict, alphak_opt_dict = buildAlphaDict(uvars_opt[:,0])

    m_tests = {}
    a_tests = {}
    BH = {}
    BICU = {}
    for ag in age_groups:
        BH[ag] = uk_opt_dict[ag]['BounceH_g']
        BICU[ag] = uk_opt_dict[ag]['BounceICU_g']
        m_tests[ag] = uk_opt_dict[ag]['Nmtest_g']
        a_tests[ag] = uk_opt_dict[ag]['Natest_g']

    # take one time step in dynamical system
    #print("Calling Time Step at The end of the for loop")
    if(use_bounce_var):
        dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict, BH, BICU)
    else:
        dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict)

    # update uhat_sequence
    uhat_seq = uvars_opt[:,1:]
