# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 16:08:59 2020

@author: Dan
"""

import os.path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
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
        'days': 5,
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
K_mtest = dynModel.parameters['global-parameters']['C_mtest']
K_atest = dynModel.parameters['global-parameters']['C_atest']



# ##############################################################################
# Testing the construction of a typical LP

# some boolean flags for running the heuristic
use_bounce_var = False   # whether to use the optimal bounce variables when forecasting the new X_hat
bounce_existing = True   # whether to allow bouncing existing patients

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

    # calculate state trajectory X_hat
    Xhat_seq, new_uhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq, use_bounce_var)
    assert( np.shape(Xhat_seq) == (Xt_dim,T-k) )
    assert( np.shape(new_uhat_seq) == (ut_dim,T-k) )
    
    # overwrite uhat with the updated one (with new bounce variables)
    #print("\nOld uhat at 1:")
    #print(uhat_seq[:,1])
    #print("\nNew uhat at 1")
    #print(new_uhat_seq[:,1])

    uhat_seq = new_uhat_seq

    ICUidx_all = slice(SEIR_groups.index('ICU_g'),Xt_dim,num_compartments)
    # print("Total people in ICU at start of period k: {}".format(np.sum(Xhat_seq[ICUidx_all,0])))
    
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
    


# ##############################################################################
# PLOT AND PRINT RESULTS
# Retrieve optimization output
u_all_opt_dict = {}
alpha_all_opt_dict = {}
for t in range(T):
    u_all_opt_dict[t], alpha_all_opt_dict[t] = buildAlphaDict(uopt_seq[:,t])
    
# Retrieve optimal lockdown decisions
# Express as dictionary where given an age group, an activity key corresponds to an np.array of length T. 
# That array holds the optimal lockdown decisions for that age group and that activity.
lockdowns_opt = {}
for ag in age_groups:
    lockdowns_opt[ag] = {}
    for act in activities:
        lockdowns_opt[ag][act] = np.zeros(T)
        for t in range(T):
            lockdowns_opt[ag][act][t] = alpha_all_opt_dict[t][ag][act]

# Retrieve optimal testing decisions
m_tests_slice = uopt_seq[controls.index('Nmtest_g'): len(uopt_seq): num_controls]
a_tests_slice = uopt_seq[controls.index('Natest_g'): len(uopt_seq): num_controls]

# # Retrieve optimal bouncing decisions
# BH_slice = uopt_seq[controls.index('BounceH_g'): len(uopt_seq): num_controls]
# BICU_slice = uopt_seq[controls.index('BounceICU_g'): len(uopt_seq): num_controls]

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
    plt.plot(time_axis_controls,m_tests_slice[i,:], label = "M tests")
    plt.plot(time_axis_controls,a_tests_slice[i,:], label = "A tests")
    # plt.plot(range(0,int(simulation_params['time_periods'])),
    # np.array(re_change_order(m_tests_vec)[group])+max(float(args.m_tests),float(args.a_tests))/100, label="M Tests")
    # plt.plot(range(0,int(simulation_params['time_periods'])), re_change_order(a_tests_vec)[group], label="A Tests")
    plt.ylim(-max(float(K_mtest),float(K_atest))/10,max(float(K_mtest),float(K_atest))+max(float(K_mtest),float(K_atest))/10)
    plt.legend(loc='upper right')

for i,group in enumerate(groups):
    plt.subplot(13,len(groups),i+1+len(groups)*7)
    plt.plot(time_axis_controls, lockdowns_opt[group]["home"]+0.01, label="Home")
    plt.plot(time_axis_controls, lockdowns_opt[group]["work"]+0.01*2, label="Work")
    plt.plot(time_axis_controls, lockdowns_opt[group]["transport"]+0.01*3, label="Transport")
    plt.plot(time_axis_controls, lockdowns_opt[group]["school"]+0.01*4, label="School")
    plt.plot(time_axis_controls, lockdowns_opt[group]["leisure"]+0.01*5, label="Leisure")
    plt.plot(time_axis_controls, lockdowns_opt[group]["other"]+0.01*6, label="Other")
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
figure.suptitle('Region: %s, Linearization Heuristic with Total Days: %s, Initial Infected percentage: %s, M-test daily capacity: %s, A-test daily capacity: %s, Mixing: %s'%(simulation_params['region'],T,perc_infected,K_mtest,K_atest,mixing_method["name"]), fontsize=22)
plt.savefig("../results_runs/optimization/"+simulation_params['region']+"_linearization_heuristic"+"_n_days_"+str(T)+"_initial_infected_percentage_"+str(perc_infected)+"_m_tests_"+str(dynModel.parameters['global-parameters']['C_mtest'])+"_a_tests_"+str(dynModel.parameters['global-parameters']['C_atest'])+"_mixing_"+mixing_method["name"]+".pdf")














