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
from linearization import *
# from forecasting_heuristic import *
import math
import pprint
import time

def run_linearization_heuristic(simulation_params):

    start_time = time.time()
    
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = 4 + 2*num_age_groups + num_age_groups*num_activities
    
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
    with open("initialization/patient_zero.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        initialization = yaml.load(file, Loader=yaml.FullLoader)

    # Update initialization
    # Put exactly initial_infected infected individuals in age group 40-49. No infected individuals in other groups.
    initialization["age_group_40_49"]["I"] = initialization["age_group_40_49"]["I"] + int(simulation_params['initial_infected_count'])
    initialization["age_group_40_49"]["S"] = initialization["age_group_40_49"]["S"] - int(simulation_params['initial_infected_count'])
    
    dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], num_time_periods, mixing_method)
    
    # add parameters for testing capacity
    dynModel.parameters['global-parameters']['C_mtest'] = simulation_params['mtest_cap']
    dynModel.parameters['global-parameters']['C_atest'] = simulation_params['atest_cap']
    
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

        assert( np.shape(Xhat_seq) == (Xt_dim,T-k) )
        assert( np.shape(new_uhat_seq) == (ut_dim,T-k) )
    
        uhat_seq = new_uhat_seq
    
        ICUidx_all = slice(SEIR_groups.index('ICU_g'),Xt_dim,num_compartments)
    
        # calculate objective parameters d, e
        D,E = calculate_objective_time_dependent_coefs(dynModel, k, Xhat_seq, uhat_seq)
    
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
        # mod.setParam( 'OutputFlag', False )     # make Gurobi silent
        mod.Params.DualReductions = 0  # change this to get explicit infeasible or unbounded
    
        # add all decisions using matrix format, and also specify objective coefficients
        obj_vec = np.reshape(obj_coefs, (ut_dim*(T-k),), 'F')  # reshape by reading along rows first
        u_vars_vec = mod.addMVar( np.shape(obj_vec), obj=obj_vec, name="u")
    
        # Sense -1 indicates a maximization problem
        mod.ModelSense = -1
    
        x_feas = np.zeros( (len(obj_vec),) )  # a feasible solution
    
        for t in range(k,T):
            #print("Time %d number of constraints %d" %(t,len(constr_coefs[t])))
            for con in range(num_constraints):
                cons_vec = np.reshape(constr_coefs[t][con], (len(obj_vec),), 'F')
                cname = ("%s[t=%d]" %(all_labels[con],t))
                mod.addConstr( u_vars_vec @ cons_vec + constr_consts[t][con] <= K[con,t], name=cname)
    
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
        if(use_bounce_var):
            dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict, BH, BICU)
        else:
            dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict)
    
        # update uhat_sequence
        uhat_seq = uvars_opt[:,1:]
    
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
    
simulation_params_l_nl_heuristic = {
    'dt':1.0,
    'region': "Ile-de-France",
    'quar_freq': 1,
    'num_daysToLockDown' : 10,
    'num_daysAfterLockDown' : 10,
    'initial_infected_count' : 1,
    'mixing_method' : {
        "name":"mult",
        "param_alpha":1.0,
        "param_beta":0.5,},
    'mtest_cap' : 100,
    'atest_cap' : 100,
    'testing_policy' : "homogeneous",
    'work_full_lockdown_factor' : 0.24,
}

simulation_params_linearization = {
    'dt':1.0,
    'region': "Ile-de-France",
    'quar_freq': 1,
    'num_days' : 20,
    'initial_infected_count' : 1,
    'mixing_method' : {
        "name":"mult",
        "param_alpha":1.0,
        "param_beta":0.5,},
    'mtest_cap' : 100,
    'atest_cap' : 100,
    'work_full_lockdown_factor' : 0.24,
}

run_nl_l_heuristic(simulation_params_l_nl_heuristic)
run_linearization_heuristic(simulation_params_linearization)