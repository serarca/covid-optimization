# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:08:59 2020

@author: Dan
"""

import os.path
import sys
import numpy as np
from inspect import getsourcefile
import time

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/heuristics")

from linearization import *
from heuristics import *

start_time = time.time()

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
perc_infected = 6
# Move population to infected (without this there is no epidem.)
for group in initialization:
	change = initialization[group]["S"]*perc_infected/100
	initialization[group]["S"] = initialization[group]["S"] - change
	initialization[group]["I"] = initialization[group]["I"] + change

# Create environment
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
#print(dynModel.time_steps)

# add parameters for testing capacity
dynModel.parameters['global-parameters']['C_mtest'] = 10000
dynModel.parameters['global-parameters']['C_atest'] = 10000

# Set up testing decisions: no testing for now
a_tests_vec, m_tests_vec = no_tests(dynModel)
#a_tests_vec, m_tests_vec = homogeneous(dynModel, [dynModel.parameters['global-parameters']['C_mtest']]*dynModel.time_steps,\
#                                       [dynModel.parameters['global-parameters']['C_atest']]*dynModel.time_steps)
tests = {
    'a_tests_vec':a_tests_vec,
    'm_tests_vec':m_tests_vec,
}

# ##############################################################################
# Testing the construction of a function that optimizes bouncing vars

# shorthand for a few useful parameters
T = dynModel.time_steps

###################
# do one run with usual bouncing and using true population at time t
dynModel.simulate(m_tests_vec, a_tests_vec, alphas_vec)
opt_val_no_bouncing_true = dynModel.get_total_reward()
dynModel.reset_time(0)

###################
# create empty model
mod = gb.Model("Optimize Bouncing Variables")
# mod.setParam( 'OutputFlag', False )     # make Gurobi silent
mod.Params.DualReductions = 0  # change this to get explicit infeasible or unbounded

# dictionary to store all bounce variables (for each period)
bounce_H = {}
bounce_ICU = {}

for t in range(T):

    #print('\n\nAdding decisions and constraints in period {}'.format(t))

    # H: add a decision variable for each age group in period t
    bounce_H[t] = mod.addVars(age_groups, name=("BH[{}]".format(t)) )
    
    # ICU: add a decision variable for each age group in period t
    bounce_ICU[t] = mod.addVars(age_groups, name=("ICU[{}]".format(t)) )
    
    # for every group in the dynamic model
    for g in dynModel.groups:
        # get flow in H
        flow_H = dynModel.groups[g].flow_H()
        # add constraint on bounce_H
        mod.addConstr( bounce_H[t][g] <= flow_H, name=("BH_{}_[{}]".format(g,t)) )
        
        # get flow in ICU
        flow_ICU = dynModel.groups[g].flow_ICU()
        # add constraint on bounce_ICU
        mod.addConstr( bounce_ICU[t][g] <= flow_ICU, name=("BICU_{}_[{}]".format(g,t)) )

    # take a step in the dynamic model using the bounce variables
    dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], alphas_vec[t], bounce_H[t], bounce_ICU[t])

    # get the state from dynModel
    Xt_plus_1 = dynModel.get_state(t)
    
    # add constraint on H capacity
    mod.addConstr( gb.quicksum(Xt_plus_1[ag]["H"] for ag in age_groups) <= dynModel.beds, name=("H_capacity_[{}]".format(t)) )
    
    # add constraint on ICU capacity
    mod.addConstr( gb.quicksum(Xt_plus_1[ag]["ICU"] for ag in age_groups) <= dynModel.icus, name=("ICU_capacity_[{}]".format(t)) )

mod.setObjective(dynModel.get_total_reward(), gb.GRB.MAXIMIZE)

# write the LP to a file
mod.write("LP_optimize_bouncing.lp")

print("Optimizing Model")
mod.optimize()

if( mod.Status ==  gb.GRB.INFEASIBLE ):
    # model was infeasible
    mod.computeIIS()  # irreducible system of infeasible inequalities
    mod.write("LP_optimize_bouncing.ilp")
    print("ERROR. Problem infeasible at time k={}. Halting...".format(k))
    assert(False)
else:
    opt_val_LP = mod.objval

#########################################################################################
# now with usual bouncing, but keeping the approximation of the population (b/c the boolean flag for gurobi in dynModel is still True)
dynModel.reset_time(0)
dynModel.simulate(m_tests_vec, a_tests_vec, alphas_vec)
opt_val_no_bouncing_approx_pop = dynModel.get_total_reward()

###################
# print all results
print("\n{}\nRunnning the simulation with proportional bouncing and true population:\n{}.".format('*'*100,opt_val_no_bouncing_true))
print("\n\nOptimal value when optimizing bouncing:\n{}.".format(opt_val_LP))
print("\n\nRunnning the simulation with proportional bouncing and population at t=0:\n{}.".format(opt_val_no_bouncing_approx_pop))