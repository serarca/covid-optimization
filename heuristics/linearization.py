# -*- coding: utf-8 -*-
import yaml
from inspect import getsourcefile
import os.path
import sys

from threadpoolctl import threadpool_limits
import numpy as np

threadpool_limits(limits=1, user_api='blas')

import pandas as pd
import math
import gurobipy as gb
__gurobi_threads = 1

from time import time
import logging
import random


current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from group import *
from simple_heuristics import no_tests

use_gpu = False
if use_gpu:
    import torch

############### PROFILING CODE ##################

def log_execution_time(function):
    def timed(*args, **kw):
        time_start = time()
        return_value = function(*args, **kw)
        time_end = time()

        execution_time = time_end - time_start

        message = f'{function.__name__}, {execution_time}'
        logging.critical(message)

        return return_value

    return timed
##################################################


age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29','age_group_30_39', 'age_group_40_49', 'age_group_50_59', 'age_group_60_69', 'age_group_70_79','age_group_80_plus']
# age_groups = ["all_age_groups"]
SEIR_groups = [ 'S_g', 'E_g', 'I_g', 'R_g', 'N_g', 'Ia_g', 'Ips_g', \
       'Ims_g', 'Iss_g', 'Rq_g', 'H_g', 'ICU_g', 'D_g' ]
activities = ['home','leisure','other','school','transport','work']
controls = [ 'Nmtest_g', 'Natest_g', 'BounceH_g', 'BounceICU_g' ]
controls.extend(activities)

num_age_groups = len(age_groups)
num_compartments = len(SEIR_groups)
num_controls = len(controls)
num_activities = len(activities)

Xt_dim = num_compartments * num_age_groups
ut_dim = num_controls * num_age_groups


numpyArrayDatatype = np.float64
# debugging = True


#def get_index_X(ag_name, SEIRg_name, age_groups, SEIR_groups):
#    return age_groups.index(ag_name)*len(SEIR_groups) + SEIR_groups.index(SEIRg_name)

#def get_index_u(ag_name, SEIRg_name, age_groups, SEIR_groups):
#    return age_groups.index(ag_name)*len(SEIR_groups) + SEIR_groups.index(SEIRg_name)

####################################
# Calculate the Jacobian with respect to X (states)
# @profile
def get_Jacobian_X(dynModel, X_hat, u_hat, mixing_method, t):
    """ Calculates the Jacobian for a given control trajectory u_hat and corresponding state trajectory X_hat """
    # This assumes that the order of components in X_hat and u_hat is as follows:
    # For X_hat: all the SEIR states for age group 1, then all the SEIR states for age group 2, etc.
    # For u_hat: all the controls for age group 1, then all the controls for age group 2, etc.
    # The order of the SEIR states is as follows:
    # X_hat = [ S_g, E_g, I_g, R_g, N_g, Ia_g, Ips_g, Ims_g, Iss_g, Rq_g, H_g, ICU_g, D_g , ...]
    # The order of the controls is as follows:
    # u_hat = [ Nmtest_g, Natest_g, BounceH_g, BounceICU_g, alpha_{g,activity1}, alpha_{g,activity2}, ... ]

    jacob = np.zeros((num_age_groups*num_compartments, num_age_groups*num_compartments), dtype=numpyArrayDatatype)

    u_hat_dict, alphas = buildAlphaDict(u_hat)

    for ag in range(0, num_age_groups):
        # Get all the useful indices for the columns
        Sg_idx = ag*num_compartments + SEIR_groups.index('S_g')
        Eg_idx = ag*num_compartments + SEIR_groups.index('E_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Ng_idx = ag*num_compartments + SEIR_groups.index('N_g')
        Rg_idx = ag*num_compartments + SEIR_groups.index('R_g')
        Iag_idx = ag*num_compartments + SEIR_groups.index('Ia_g')
        Ipsg_idx = ag*num_compartments + SEIR_groups.index('Ips_g')
        Imsg_idx = ag*num_compartments + SEIR_groups.index('Ims_g')
        Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        Nmtestg_idx = ag*num_controls + controls.index('Nmtest_g')
        Natestg_idx = ag*num_controls + controls.index('Natest_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')

        # These are subarrays that only contain I_h, Rq_h, N_h out of X_hat
        I_h_slice = X_hat[SEIR_groups.index('I_g'): len(X_hat): num_compartments]
        Rq_h_slice = X_hat[SEIR_groups.index('Rq_g'): len(X_hat): num_compartments]
        N_h_slice = X_hat[SEIR_groups.index('N_g'): len(X_hat): num_compartments]

        infection_prob_arr = np.divide(I_h_slice, (N_h_slice + Rq_h_slice))
        c_gh_arr = calcContacts(dynModel, alphas, mixing_method, ag)

        contacts = np.dot(c_gh_arr, infection_prob_arr)

        ########### f^Sg
        # df^Sg/dSg
        jacob[Sg_idx,Sg_idx] = - dynModel.groups[age_groups[ag]].parameters['beta'][t] * contacts

        # df^Sg/dNh all h
        # for ah in range(0, num_age_groups):
        #     jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('N_g')] = dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr[ah] * I_h_slice[ah] / ((N_h_slice[ah] + Rq_h_slice[ah])**2)
        jacob[Sg_idx, SEIR_groups.index('N_g')::num_compartments] = dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr * I_h_slice / ((N_h_slice + Rq_h_slice)**2)

        # df^Sg/dIh all h
        # for ah in range(0, num_age_groups):
        #     jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('I_g')] = - dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr[ah] / (N_h_slice[ah] + Rq_h_slice[ah])
        jacob[Sg_idx, SEIR_groups.index('I_g')::num_compartments] = - dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr / (N_h_slice + Rq_h_slice)

        # df^Sg/dRqh all h
        # for ah in range(0, num_age_groups):
        #     jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')] = dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr[ah] * I_h_slice[ah] / ((N_h_slice[ah] + Rq_h_slice[ah])**2)
        jacob[Sg_idx,SEIR_groups.index('Rq_g')::num_compartments] = dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr * I_h_slice / ((N_h_slice + Rq_h_slice)**2)

        ########### f^Eg
        # df^Eg/dSg
        jacob[Eg_idx,Sg_idx] = - jacob[Sg_idx,Sg_idx]

        # df^Eg/dNh all h
        # for ah in range(num_age_groups):
        #     jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('N_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('N_g')]
        jacob[Eg_idx,SEIR_groups.index('N_g')::num_compartments] = - jacob[Sg_idx,SEIR_groups.index('N_g')::num_compartments]

        # df^Eg/dIh all h
        # for ah in range(num_age_groups):
        #     jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('I_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('I_g')]
        jacob[Eg_idx,SEIR_groups.index('I_g')::num_compartments] = - jacob[Sg_idx,SEIR_groups.index('I_g')::num_compartments]

        # df^Eg/dRqh all h
        # for ah in range(num_age_groups):
        #     jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')]
        jacob[Eg_idx,SEIR_groups.index('Rq_g')::num_compartments] = - jacob[Sg_idx,SEIR_groups.index('Rq_g')::num_compartments]

        # # df^Eg/dEg
        jacob[Eg_idx,Eg_idx] = - dynModel.groups[age_groups[ag]].parameters['sigma']

        #### Derivatives for the function that yields I_g
        # Deriv w.r.t E_g
        jacob[Ig_idx, Eg_idx] = dynModel.groups[age_groups[ag]].parameters['sigma']
        # Deriv w.r.t I_g
        jacob[Ig_idx, Ig_idx] = -dynModel.groups[age_groups[ag]].parameters['mu'] - u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Ig_idx, Ng_idx] = u_hat[Nmtestg_idx] * X_hat[Ig_idx]/((X_hat[Ng_idx])**2)

        #### Derivatives for the function that yields R_g
        # Deriv w.r.t I_g
        jacob[Rg_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['mu']*(1 - dynModel.groups[age_groups[ag]].parameters['p_H'] - dynModel.groups[age_groups[ag]].parameters['p_ICU'])
        # Deriv w.r.t R_g
        jacob[Rg_idx, Rg_idx] = -u_hat[Natestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Rg_idx, Ng_idx] = u_hat[Natestg_idx]*X_hat[Rg_idx]/((X_hat[Ng_idx])**2)

        #### Derivatives for the function that yields Ia_g
        # Deriv w.r.t I_g
        jacob[Iag_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ia']*u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Iag_idx, Ng_idx] = -dynModel.groups[age_groups[ag]].parameters['p_Ia']*u_hat[Nmtestg_idx]*X_hat[Ig_idx]/((X_hat[Ng_idx])**2)
        # Deriv w.r.t Ia_g
        jacob[Iag_idx, Iag_idx] = -dynModel.groups[age_groups[ag]].parameters['mu']

        #### Derivatives for the function that yields Ips_g
        # Deriv w.r.t I_g
        jacob[Ipsg_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ips']*u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Ipsg_idx, Ng_idx] = -dynModel.groups[age_groups[ag]].parameters['p_Ips']*u_hat[Nmtestg_idx]*X_hat[Ig_idx]/((X_hat[Ng_idx])**2)
        # Deriv w.r.t Ips_g
        jacob[Ipsg_idx, Ipsg_idx] = -dynModel.groups[age_groups[ag]].parameters['mu']

        #### Derivatives for the function that yields Ims_g
        # Deriv w.r.t I_g
        jacob[Imsg_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ims']*u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Imsg_idx, Ng_idx] = -dynModel.groups[age_groups[ag]].parameters['p_Ims']*u_hat[Nmtestg_idx]*X_hat[Ig_idx]/((X_hat[Ng_idx])**2)
        # Deriv w.r.t Ims_g
        jacob[Imsg_idx, Imsg_idx] = -dynModel.groups[age_groups[ag]].parameters['mu']


        #### Derivatives for the function that yields Iss_g
        # Deriv w.r.t I_g
        jacob[Issg_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['p_Iss']*u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Issg_idx, Ng_idx] = -dynModel.groups[age_groups[ag]].parameters['p_Iss']*u_hat[Nmtestg_idx]*X_hat[Ig_idx]/((X_hat[Ng_idx])**2)
        # Deriv w.r.t Iss_g
        jacob[Issg_idx, Issg_idx] = -dynModel.groups[age_groups[ag]].parameters['mu']

        #### Derivatives for the function that yields Rq_g
        # Deriv w.r.t Ia, Ips, Iss
        jacob[Rqg_idx, Iag_idx] = dynModel.groups[age_groups[ag]].parameters['mu']
        jacob[Rqg_idx, Ipsg_idx] = dynModel.groups[age_groups[ag]].parameters['mu']
        jacob[Rqg_idx, Imsg_idx] = dynModel.groups[age_groups[ag]].parameters['mu']
        # Deriv w.r.t H_g
        jacob[Rqg_idx, Hg_idx] = dynModel.groups[age_groups[ag]].parameters['lambda_H_R']
        # Deriv w.r.t ICU_g
        jacob[Rqg_idx, ICUg_idx] = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_R']
        # Deriv w.r.t R_g
        jacob[Rqg_idx, Rg_idx] = u_hat[Natestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Rqg_idx, Ng_idx] = -u_hat[Natestg_idx]*X_hat[Rg_idx]/((X_hat[Ng_idx])**2)

        ########### f^Ng
        # df^Ng/dNg
        jacob[Ng_idx,Ng_idx] = u_hat[Nmtestg_idx] * X_hat[Ig_idx] / (X_hat[Ng_idx]**2) + u_hat[Natestg_idx] * X_hat[Rg_idx] / (X_hat[Ng_idx]**2)

        # df^Ng/dIg
        jacob[Ng_idx,Ig_idx] = - u_hat[Nmtestg_idx] / X_hat[Ng_idx] - dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_H'] + dynModel.groups[age_groups[ag]].parameters['p_ICU'])

        # df^Ng/dRg
        jacob[Ng_idx,Rg_idx] = - u_hat[Natestg_idx] / X_hat[Ng_idx]

        ########### f^Hg
        # df^Hg/dHg
        jacob[Hg_idx,Hg_idx] = - (dynModel.groups[age_groups[ag]].parameters['lambda_H_R'] + dynModel.groups[age_groups[ag]].parameters['lambda_H_D'])
        # df^Hg/dIg
        jacob[Hg_idx,Ig_idx] = dynModel.groups[age_groups[ag]].parameters['mu'] * dynModel.groups[age_groups[ag]].parameters['p_H']
        # df^Hg/dIssg
        jacob[Hg_idx,Issg_idx] = dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_H'] / (dynModel.groups[age_groups[ag]].parameters['p_H'] + dynModel.groups[age_groups[ag]].parameters['p_ICU']))

        ########### f^ICUg
        # df^ICUg/dICUg
        jacob[ICUg_idx,ICUg_idx] = - (dynModel.groups[age_groups[ag]].parameters['lambda_ICU_R'] + dynModel.groups[age_groups[ag]].parameters['lambda_ICU_D'])
        # df^ICUg/dIg
        jacob[ICUg_idx,Ig_idx] = dynModel.groups[age_groups[ag]].parameters['mu'] * dynModel.groups[age_groups[ag]].parameters['p_ICU']
        # df^ICUg/dIssg
        jacob[ICUg_idx,Issg_idx] = dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_ICU'] / (dynModel.groups[age_groups[ag]].parameters['p_H'] + dynModel.groups[age_groups[ag]].parameters['p_ICU']))

        ########### f^Dg
        # df^Dg/dHg
        jacob[Dg_idx,Hg_idx] = dynModel.groups[age_groups[ag]].parameters['lambda_H_D']
        # df^Dg/dICUg
        jacob[Dg_idx,ICUg_idx] = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_D']

    return jacob


def get_Jacobian_X_optimized(dynModel, X_hat, u_hat, mixing_method, t):
    """ Calculates the Jacobian for a given control trajectory u_hat and corresponding state trajectory X_hat """
    # This assumes that the order of components in X_hat and u_hat is as follows:
    # For X_hat: all the SEIR states for age group 1, then all the SEIR states for age group 2, etc.
    # For u_hat: all the controls for age group 1, then all the controls for age group 2, etc.
    # The order of the SEIR states is as follows:
    # X_hat = [ S_g, E_g, I_g, R_g, N_g, Ia_g, Ips_g, Ims_g, Iss_g, Rq_g, H_g, ICU_g, D_g , ...]
    # The order of the controls is as follows:
    # u_hat = [ Nmtest_g, Natest_g, BounceH_g, BounceICU_g, alpha_{g,activity1}, alpha_{g,activity2}, ... ]

    jacob = np.zeros((num_age_groups*num_compartments, num_age_groups*num_compartments), dtype=numpyArrayDatatype)

    u_hat_dict, alphas = buildAlphaDict(u_hat)

    Sg_idx_opt = list(range(SEIR_groups.index('S_g'), num_age_groups * num_compartments, num_compartments))

    # These are subarrays that only contain I_h, Rq_h, N_h out of X_hat
    I_h_slice = X_hat[SEIR_groups.index('I_g'):: num_compartments]
    Rq_h_slice = X_hat[SEIR_groups.index('Rq_g'):: num_compartments]
    N_h_slice = X_hat[SEIR_groups.index('N_g'):: num_compartments]

    infection_prob_arr = np.divide(I_h_slice, (N_h_slice + Rq_h_slice))
    
    c_gh_arr = np.zeros((num_age_groups,num_age_groups))

    for ag in range(0, num_age_groups):
        c_gh_arr[ag] = calcContacts(dynModel, alphas, mixing_method, ag)

    contacts = np.dot(c_gh_arr, infection_prob_arr)

    ########### f^Sg
    # df^Sg/dSg
    # jacob[Sg_idx,Sg_idx] = - dynModel.groups[age_groups[ag]].parameters['beta'][t] * contacts
    for ag in range(0, num_age_groups):
        # Get all the useful indices for the columns
        Sg_idx_prime = ag*num_compartments + SEIR_groups.index('S_g')
        jacob[Sg_idx_prime,Sg_idx_prime] = - dynModel.groups[age_groups[ag]].parameters['beta'][t] * contacts[ag]

    # df^Sg/dNh all h
    # for ah in range(0, num_age_groups):
    #     jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('N_g')] = dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr[ah] * I_h_slice[ah] / ((N_h_slice[ah] + Rq_h_slice[ah])**2)
    jacob[Sg_idx_opt, SEIR_groups.index('N_g')::num_compartments] = np.multiply(np.multiply(np.array([dynModel.groups[age_groups[ag]].parameters['beta'][t] for ag in range(num_age_groups)]) * X_hat[Sg_idx_opt], c_gh_arr.T).T, I_h_slice / ((N_h_slice + Rq_h_slice)**2))

    # df^Sg/dIh all h
    # for ah in range(0, num_age_groups):
    #     jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('I_g')] = - dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr[ah] / (N_h_slice[ah] + Rq_h_slice[ah])
    jacob[Sg_idx_opt, SEIR_groups.index('I_g')::num_compartments] = -np.multiply(np.multiply(np.array([dynModel.groups[age_groups[ag]].parameters['beta'][t] for ag in range(num_age_groups)]) * X_hat[Sg_idx_opt], c_gh_arr.T).T, 1 / (N_h_slice + Rq_h_slice))

    # df^Sg/dRqh all h
    # for ah in range(0, num_age_groups):
    #     jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')] = dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx] * c_gh_arr[ah] * I_h_slice[ah] / ((N_h_slice[ah] + Rq_h_slice[ah])**2)
    jacob[Sg_idx_opt,SEIR_groups.index('Rq_g')::num_compartments] = jacob[Sg_idx_opt, SEIR_groups.index('N_g')::num_compartments]
    
    for ag in range(0, num_age_groups):
        # Get all the useful indices for the columns
        Sg_idx = ag*num_compartments + SEIR_groups.index('S_g')
        Eg_idx = ag*num_compartments + SEIR_groups.index('E_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Ng_idx = ag*num_compartments + SEIR_groups.index('N_g')
        Rg_idx = ag*num_compartments + SEIR_groups.index('R_g')
        Iag_idx = ag*num_compartments + SEIR_groups.index('Ia_g')
        Ipsg_idx = ag*num_compartments + SEIR_groups.index('Ips_g')
        Imsg_idx = ag*num_compartments + SEIR_groups.index('Ims_g')
        Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        Nmtestg_idx = ag*num_controls + controls.index('Nmtest_g')
        Natestg_idx = ag*num_controls + controls.index('Natest_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')
       
        ########### f^Eg
        # df^Eg/dSg
        jacob[Eg_idx, Sg_idx] = - jacob[Sg_idx, Sg_idx] 

        # df^Eg/dNh all h
        # for ah in range(num_age_groups):
        #     jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('N_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('N_g')]
        jacob[Eg_idx,SEIR_groups.index('N_g')::num_compartments] = - jacob[Sg_idx,SEIR_groups.index('N_g')::num_compartments]

        # df^Eg/dIh all h
        # for ah in range(num_age_groups):
        #     jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('I_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('I_g')]
        jacob[Eg_idx,SEIR_groups.index('I_g')::num_compartments] = - jacob[Sg_idx,SEIR_groups.index('I_g')::num_compartments]

        # df^Eg/dRqh all h
        # for ah in range(num_age_groups):
        #     jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')]       
        jacob[Eg_idx,SEIR_groups.index('Rq_g')::num_compartments] = - jacob[Sg_idx,SEIR_groups.index('Rq_g')::num_compartments]

        # # df^Eg/dEg
        jacob[Eg_idx, Eg_idx] = -dynModel.groups[age_groups[ag]].parameters['sigma']

        #### Derivatives for the function that yields I_g
        # Deriv w.r.t E_g
        # t.append(time())
        jacob[Ig_idx, Eg_idx] = dynModel.groups[age_groups[ag]].parameters['sigma']

        # Deriv w.r.t I_g
        jacob[Ig_idx, Ig_idx] = -dynModel.groups[age_groups[ag]].parameters['mu'] - u_hat[Nmtestg_idx]/X_hat[Ng_idx]
            
        # Deriv w.r.t N_g
        jacob[Ig_idx, Ng_idx] = u_hat[Nmtestg_idx] * X_hat[Ig_idx]/((X_hat[Ng_idx])**2)

        #### Derivatives for the function that yields R_g
        # Deriv w.r.t I_g
        # t.append(time())
        jacob[Rg_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['mu']*(1 - dynModel.groups[age_groups[ag]].parameters['p_H'] - dynModel.groups[age_groups[ag]].parameters['p_ICU'])
        # Deriv w.r.t R_g
        jacob[Rg_idx, Rg_idx] = -u_hat[Natestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Rg_idx, Ng_idx] = u_hat[Natestg_idx]*X_hat[Rg_idx]/((X_hat[Ng_idx])**2)

        #### Derivatives for the function that yields Ia_g
        # Deriv w.r.t I_g
        # t.append(time())
        jacob[Iag_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ia']*u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Iag_idx, Ng_idx] = -dynModel.groups[age_groups[ag]].parameters['p_Ia']*u_hat[Nmtestg_idx]*X_hat[Ig_idx]/((X_hat[Ng_idx])**2)
        # Deriv w.r.t Ia_g
        jacob[Iag_idx, Iag_idx] = -dynModel.groups[age_groups[ag]].parameters['mu']

        #### Derivatives for the function that yields Ips_g
        # Deriv w.r.t I_g
        # t.append(time())
        jacob[Ipsg_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ips']*u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Ipsg_idx, Ng_idx] = -dynModel.groups[age_groups[ag]].parameters['p_Ips']*u_hat[Nmtestg_idx]*X_hat[Ig_idx]/((X_hat[Ng_idx])**2)
        # Deriv w.r.t Ips_g
        jacob[Ipsg_idx, Ipsg_idx] = -dynModel.groups[age_groups[ag]].parameters['mu']

        #### Derivatives for the function that yields Ims_g
        # Deriv w.r.t I_g
        # t.append(time())
        jacob[Imsg_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ims']*u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Imsg_idx, Ng_idx] = -dynModel.groups[age_groups[ag]].parameters['p_Ims']*u_hat[Nmtestg_idx]*X_hat[Ig_idx]/((X_hat[Ng_idx])**2)
        # Deriv w.r.t Ims_g
        jacob[Imsg_idx, Imsg_idx] = -dynModel.groups[age_groups[ag]].parameters['mu']

        #### Derivatives for the function that yields Iss_g
        # Deriv w.r.t I_g
        # t.append(time())
        jacob[Issg_idx, Ig_idx] = dynModel.groups[age_groups[ag]].parameters['p_Iss']*u_hat[Nmtestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Issg_idx, Ng_idx] = -dynModel.groups[age_groups[ag]].parameters['p_Iss']*u_hat[Nmtestg_idx]*X_hat[Ig_idx]/((X_hat[Ng_idx])**2)
        # Deriv w.r.t Iss_g
        jacob[Issg_idx, Issg_idx] = -dynModel.groups[age_groups[ag]].parameters['mu']

        #### Derivatives for the function that yields Rq_g
        # Deriv w.r.t Ia, Ips, Iss
        # t.append(time())
        jacob[Rqg_idx, Iag_idx] = dynModel.groups[age_groups[ag]].parameters['mu']
        jacob[Rqg_idx, Ipsg_idx] = dynModel.groups[age_groups[ag]].parameters['mu']
        jacob[Rqg_idx, Imsg_idx] = dynModel.groups[age_groups[ag]].parameters['mu']
        # Deriv w.r.t H_g
        jacob[Rqg_idx, Hg_idx] = dynModel.groups[age_groups[ag]].parameters['lambda_H_R']
        # Deriv w.r.t ICU_g
        jacob[Rqg_idx, ICUg_idx] = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_R']
        # Deriv w.r.t R_g
        jacob[Rqg_idx, Rg_idx] = u_hat[Natestg_idx]/X_hat[Ng_idx]
        # Deriv w.r.t N_g
        jacob[Rqg_idx, Ng_idx] = -u_hat[Natestg_idx]*X_hat[Rg_idx]/((X_hat[Ng_idx])**2)

        ########### f^Ng
        # df^Ng/dNg
        # t.append(time())
        jacob[Ng_idx,Ng_idx] = u_hat[Nmtestg_idx] * X_hat[Ig_idx] / (X_hat[Ng_idx]**2) + u_hat[Natestg_idx] * X_hat[Rg_idx] / (X_hat[Ng_idx]**2)

        # df^Ng/dIg
        jacob[Ng_idx,Ig_idx] = - u_hat[Nmtestg_idx] / X_hat[Ng_idx] - dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_H'] + dynModel.groups[age_groups[ag]].parameters['p_ICU'])

        # df^Ng/dRg
        jacob[Ng_idx,Rg_idx] = - u_hat[Natestg_idx] / X_hat[Ng_idx]

        ########### f^Hg
        # t.append(time())
        # df^Hg/dHg
        jacob[Hg_idx,Hg_idx] = - (dynModel.groups[age_groups[ag]].parameters['lambda_H_R'] + dynModel.groups[age_groups[ag]].parameters['lambda_H_D'])
        # df^Hg/dIg
        jacob[Hg_idx,Ig_idx] = dynModel.groups[age_groups[ag]].parameters['mu'] * dynModel.groups[age_groups[ag]].parameters['p_H']
        # df^Hg/dIssg
        jacob[Hg_idx,Issg_idx] = dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_H'] / (dynModel.groups[age_groups[ag]].parameters['p_H'] + dynModel.groups[age_groups[ag]].parameters['p_ICU']))

        ########### f^ICUg
        # df^ICUg/dICUg
        # t.append(time())
        jacob[ICUg_idx,ICUg_idx] = - (dynModel.groups[age_groups[ag]].parameters['lambda_ICU_R'] + dynModel.groups[age_groups[ag]].parameters['lambda_ICU_D'])
        # df^ICUg/dIg
        jacob[ICUg_idx,Ig_idx] = dynModel.groups[age_groups[ag]].parameters['mu'] * dynModel.groups[age_groups[ag]].parameters['p_ICU']
        # df^ICUg/dIssg
        jacob[ICUg_idx,Issg_idx] = dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_ICU'] / (dynModel.groups[age_groups[ag]].parameters['p_H'] + dynModel.groups[age_groups[ag]].parameters['p_ICU']))

        ########### f^Dg
        # t.append(time())
        # df^Dg/dHg
        jacob[Dg_idx,Hg_idx] = dynModel.groups[age_groups[ag]].parameters['lambda_H_D']
        # df^Dg/dICUg
        jacob[Dg_idx,ICUg_idx] = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_D']

    return jacob


####################################
# Calculate the Jacobian with respect to u (decisions/controls)
# @profile
def get_Jacobian_u(dynModel, X_hat, u_hat, mixing_method, t):
    """ Calculates the Jacobian with respect to decisions, for a given control trajectory u_hat and corresponding state trajectory X_hat """
    # For now, for mult mixing this is done for the model with powers ell_g^alpha * ell_h^beta
    # This assumes that the order of components in X_hat and u_hat is as follows:
    # For X_hat: all the SEIR states for age group 1, then all the SEIR states for age group 2, etc.
    # For u_hat: all the controls for age group 1, then all the controls for age group 2, etc.
    # The order of the SEIR states is as follows:
    # X_hat = [ S_g, E_g, I_g, R_g, N_g, Ia_g, Ips_g, Ims_g, Iss_g, Rq_g, H_g, ICU_g, D_g , ...]
    # The order of the controls is as follows:
    # u_hat = [ Nmtest_g, Natest_g, BounceH_g, BounceICU_g, alpha_{g,activity1}, alpha_{g,activity2}, ... ]

    alpha = mixing_method['param_alpha']
    beta = mixing_method['param_beta']

    jacob = np.zeros((num_age_groups*num_compartments,num_age_groups*num_controls), dtype=numpyArrayDatatype)

    I_h_slice = X_hat[SEIR_groups.index('I_g'): len(X_hat): num_compartments]
    Rq_h_slice = X_hat[SEIR_groups.index('Rq_g'): len(X_hat): num_compartments]
    N_h_slice = X_hat[SEIR_groups.index('N_g'): len(X_hat): num_compartments]
    rho_array = np.divide(I_h_slice, (N_h_slice + Rq_h_slice)) # Models the probability of contact with an infected individual

    for ag in range(0,num_age_groups):
        # Get all the useful indices for the columns
        Sg_idx = ag*num_compartments + SEIR_groups.index('S_g')
        Eg_idx = ag*num_compartments + SEIR_groups.index('E_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Ng_idx = ag*num_compartments + SEIR_groups.index('N_g')
        Rg_idx = ag*num_compartments + SEIR_groups.index('R_g')
        Iag_idx = ag*num_compartments + SEIR_groups.index('Ia_g')
        Ipsg_idx = ag*num_compartments + SEIR_groups.index('Ips_g')
        Imsg_idx = ag*num_compartments + SEIR_groups.index('Ims_g')
        Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')

        Nmtestg_idx = ag*num_controls + controls.index('Nmtest_g')
        Natestg_idx = ag*num_controls + controls.index('Natest_g')
        BounceHg_idx = ag*num_controls + controls.index('BounceH_g')
        BounceICUg_idx = ag*num_controls + controls.index('BounceICU_g')

        # M tests and A tests
            # N:
        jacob[Ng_idx , Nmtestg_idx] = -X_hat[Ig_idx]/X_hat[Ng_idx]
        jacob[Ng_idx , Natestg_idx] = -X_hat[Rg_idx]/X_hat[Ng_idx]

            # I:
        jacob[Ig_idx,Nmtestg_idx] = -X_hat[Ig_idx]/X_hat[Ng_idx]

            # R:
        jacob[Rg_idx,Natestg_idx] = -X_hat[Rg_idx]/X_hat[Ng_idx]

            # I_j^q:
        jacob[Iag_idx,Nmtestg_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ia']*X_hat[Ig_idx]/X_hat[Ng_idx]
        jacob[Ipsg_idx,Nmtestg_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ips']*X_hat[Ig_idx]/X_hat[Ng_idx]
        jacob[Imsg_idx,Nmtestg_idx] = dynModel.groups[age_groups[ag]].parameters['p_Ims']*X_hat[Ig_idx]/X_hat[Ng_idx]
        jacob[Issg_idx,Nmtestg_idx] = dynModel.groups[age_groups[ag]].parameters['p_Iss']*X_hat[Ig_idx]/X_hat[Ng_idx]

            # R^q:
        jacob[Rqg_idx,Natestg_idx] = X_hat[Rg_idx]/X_hat[Ng_idx]


        # Bouncing variables
            # H:
        jacob[Hg_idx,BounceHg_idx] = -1

            # ICU:
        jacob[ICUg_idx,BounceICUg_idx] = -1

            # D:
        jacob[Dg_idx,BounceHg_idx] = 1
        jacob[Dg_idx,BounceICUg_idx] = 1

        # Lockdown alpha's
        for act in range(num_controls-num_activities,num_controls):
            partial_contacts_g_array = np.zeros(num_age_groups, dtype=numpyArrayDatatype)

            lga_idx = ag * num_controls + act
            lga = u_hat[lga_idx]

            offset = num_controls-num_activities

            for ah in range(0,num_age_groups):

                Ih_idx = ah*num_compartments + SEIR_groups.index('I_g')
                Nh_idx = ah*num_compartments + SEIR_groups.index('N_g')
                Rqh_idx = ah*num_compartments + SEIR_groups.index('Rq_g')

                lha_idx = ah * num_controls + act
                lha = u_hat[lha_idx]

                if (mixing_method['name']=="mult"):
                    if ah != ag:
                        partial_contacts_g_array[ah] = alpha*dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]]\
                            *(0 if lga==0 else lga**(alpha-1))*(lha**(beta))
                        partial_contacts_h = beta*dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]]\
                            *(lga**(alpha))*(0 if lha==0 else lha**(beta-1))
                    else:
                        partial_contacts_g_array[ah] = (alpha+beta)*dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]]\
                            *(0 if lga==0 else lga**(alpha-1))*(lha**(beta))
                        partial_contacts_h = (alpha+beta)*dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]]\
                            *(lga**(alpha))*(0 if lha==0 else lha**(beta-1))
                if (mixing_method['name']=="maxmin"):
                    explga = math.exp(alpha * lga)
                    explha = math.exp(alpha * lha)
                    if ah != ag:
                        partial_contacts_g_array[ah] = dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]] \
                            * ((alpha * lga * explga * explha - alpha * explga * lha * explha + explga * explha + explga**2)/ ((explga + explha)**2))
                        partial_contacts_h = dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]] \
                            * ((alpha * lha * explha * explga - alpha * explha * lga * explga + explha * explga + explha**2)/ ((explga + explha)**2))
                    else:
                        partial_contacts_g_array[ah] = dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]]
                        partial_contacts_h = dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]]

                # S:
                jacob[Sg_idx,lha_idx] = -dynModel.groups[age_groups[ag]].parameters['beta'][t] * X_hat[Sg_idx]*partial_contacts_h\
                    *X_hat[Ih_idx]/(X_hat[Nh_idx]+X_hat[Rqh_idx])
                # E:
                jacob[Eg_idx,lha_idx] = - jacob[Sg_idx,lha_idx]

            # S:
            jacob[Sg_idx,lga_idx] = -dynModel.groups[age_groups[ag]].parameters['beta'][t]*X_hat[Sg_idx]*np.dot(rho_array,partial_contacts_g_array)
            # E:
            jacob[Eg_idx,lga_idx] = - jacob[Sg_idx,lga_idx]

    return jacob

####################################
# Build X_hat given a dynModel, a starting point k, and a sequence of controls u_hat
def get_X_hat_sequence(dynModel, k, u_hat_sequence, use_bounce):
    """Given a dynamical model, a starting point k, and the controls for time periods k to T-1 for tests and lockdowns, we start the dynamical model at time k, and then run it until time T-1 with the controls in u_hat.
    This produces the nominal trajectory X_hat_sequence. X_hat_sequence is a np.array of shape (num_compartments * num_age_groups, T-k), where each column represents the X_hat at time k, k+1,...
    This assumes that the dynamical model has already been run up to point k (it takes the states at time k as the starting points for the new nominal trajectory).
    We assume as well that u_hat_sequence is a 2-d numpy array with shape (num_controls * num_age_groups, T-k) with each column corresponding to a u_hat at time k, k+1,..., T. Hence, u_hat_sequence[:,k] gives the u_hat at time k.
    Note we are not using the bouncing variables in forecasting X_hat_sequence.
    Note: we return x_hat[k], x_hat[k+1], ..., x_hat[T].
    """
    #NOTE. There are a few extra arguments, as follows:
    # Argument 'use_bounce' determines whether to use bounce variables or not

    # Erase the states after k so as to reset the dyn model
    dynModel.reset_time(k)

    # The total time horizon for the dynamical model
    T = dynModel.time_steps

    X_hat_sequence = np.zeros((num_compartments * num_age_groups, T-k+1), dtype=numpyArrayDatatype)
    new_uhat_sequence = np.copy(u_hat_sequence)  # initialized with the old one

    Hidx_all = slice(SEIR_groups.index('H_g'),np.shape(X_hat_sequence)[0],num_compartments)
    ICUidx_all = slice(SEIR_groups.index('ICU_g'),np.shape(X_hat_sequence)[0],num_compartments)

    for t in range(T-k+1):

        # get the state from the dynamic model
        state = dynModel.get_state(t + k)

        # write it into X_hat
        X_hat_sequence[:,t] = dict_to_X(state)

        if t < T-k:
            # build a dictionary of decisions/controls
            u_hat_dict, alphas = buildAlphaDict(u_hat_sequence[:,t])

            #Create m and a tests in the format taken by dynModel
            m_tests = {}
            a_tests = {}
            BH = {}
            BICU = {}
            for ag in age_groups:
                BH[ag] = u_hat_dict[ag]['BounceH_g']
                BICU[ag] = u_hat_dict[ag]['BounceICU_g']
                m_tests[ag] = u_hat_dict[ag]['Nmtest_g']
                a_tests[ag] = u_hat_dict[ag]['Natest_g']

            # take one time step in dynamical system.
            if(use_bounce):
                dynModel.take_time_step(m_tests, a_tests, alphas, BH, BICU)
            else:
                dynModel.take_time_step(m_tests, a_tests, alphas)

            # get the updated bounce variables and write them in new control
            new_bounce = dynModel.get_bounce(t + k)
            for ag in range(num_age_groups):
                #print("ICU: Age group {}: bouncing {}".format(age_groups[ag],new_bounce[age_groups[ag]]['B_ICU']))
                new_uhat_sequence[ag * num_controls + controls.index('BounceH_g'),t] = new_bounce[age_groups[ag]]['B_H']
                new_uhat_sequence[ag * num_controls + controls.index('BounceICU_g'),t] = new_bounce[age_groups[ag]]['B_ICU']

    # Erase the states after k so as to reset the dyn model
    dynModel.reset_time(k)

    return X_hat_sequence, new_uhat_sequence

####################################
# builds a dictionary with decisions from a large numpy array (for a given period)
def buildAlphaDict(u_hat_array):
    """Given an array u_hat_array, builds a dictionary for all decisions that is compatible with DynModel"""
    u_hat_dict = {}
    alphas = {}
    for ag in range(0, num_age_groups):
        u_hat_dict[age_groups[ag]] = {}
        alphas[age_groups[ag]] = {}
        u_hat_dict[age_groups[ag]]['Nmtest_g'] = u_hat_array[ag * num_controls + controls.index('Nmtest_g')]
        u_hat_dict[age_groups[ag]]['Natest_g'] = u_hat_array[ag * num_controls + controls.index('Natest_g')]
        u_hat_dict[age_groups[ag]]['BounceH_g'] = u_hat_array[ag * num_controls + controls.index('BounceH_g')] if u_hat_array[ag * num_controls + controls.index('BounceH_g')] != -1 else False
        u_hat_dict[age_groups[ag]]['BounceICU_g'] = u_hat_array[ag * num_controls + controls.index('BounceICU_g')] if u_hat_array[ag * num_controls + controls.index('BounceICU_g')] != -1 else False

        alphas[age_groups[ag]]['home'] = u_hat_array[ag * num_controls + controls.index('home')]
        alphas[age_groups[ag]]['leisure'] = u_hat_array[ag * num_controls + controls.index('leisure')]
        alphas[age_groups[ag]]['other'] = u_hat_array[ag * num_controls + controls.index('other')]
        alphas[age_groups[ag]]['school'] = u_hat_array[ag * num_controls + controls.index('school')]
        alphas[age_groups[ag]]['transport'] = u_hat_array[ag * num_controls + controls.index('transport')]
        alphas[age_groups[ag]]['work'] = u_hat_array[ag * num_controls + controls.index('work')]

    return u_hat_dict, alphas


# Takes a dictionary of testing and bouncing variables, and a dictionary of alphas, and builds an array of u
def dict_to_u(u_hat_dict, alphas):

    u_hat_array = np.zeros(num_controls * num_age_groups, dtype=numpyArrayDatatype)

    for ag in range(0, num_age_groups):
        u_hat_array[ag * num_controls + controls.index('Nmtest_g')] = u_hat_dict[age_groups[ag]]['Nmtest_g']
        u_hat_array[ag * num_controls + controls.index('Natest_g')] = u_hat_dict[age_groups[ag]]['Natest_g']
        u_hat_array[ag * num_controls + controls.index('BounceH_g')] = u_hat_dict[age_groups[ag]]['BounceH_g'] if (u_hat_dict[age_groups[ag]]['BounceH_g'] is not False) else -1
        u_hat_array[ag * num_controls + controls.index('BounceICU_g')] = u_hat_dict[age_groups[ag]]['BounceICU_g'] if (u_hat_dict[age_groups[ag]]['BounceICU_g'] is not False) else -1

        u_hat_array[ag * num_controls + controls.index('home')] = alphas[age_groups[ag]]['home']
        u_hat_array[ag * num_controls + controls.index('leisure')] = alphas[age_groups[ag]]['leisure']
        u_hat_array[ag * num_controls + controls.index('other')] = alphas[age_groups[ag]]['other']
        u_hat_array[ag * num_controls + controls.index('school')] = alphas[age_groups[ag]]['school']
        u_hat_array[ag * num_controls + controls.index('transport')] = alphas[age_groups[ag]]['transport']
        u_hat_array[ag * num_controls + controls.index('work')] = alphas[age_groups[ag]]['work']

    return u_hat_array


# Converts an array of X into a dictionary that can be understood by dynModel as a state
def X_to_dict(X):
    X_dict = {}
    for ag in range(num_age_groups):
        X_dict[age_groups[ag]] = {}
        Sg_idx = ag*num_compartments + SEIR_groups.index('S_g')
        Eg_idx = ag*num_compartments + SEIR_groups.index('E_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Ng_idx = ag*num_compartments + SEIR_groups.index('N_g')
        Rg_idx = ag*num_compartments + SEIR_groups.index('R_g')
        Iag_idx = ag*num_compartments + SEIR_groups.index('Ia_g')
        Ipsg_idx = ag*num_compartments + SEIR_groups.index('Ips_g')
        Imsg_idx = ag*num_compartments + SEIR_groups.index('Ims_g')
        Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')

        X_dict[age_groups[ag]]['S'] = X[Sg_idx]
        X_dict[age_groups[ag]]['E'] = X[Eg_idx]
        X_dict[age_groups[ag]]['I'] = X[Ig_idx]
        X_dict[age_groups[ag]]['R'] = X[Rg_idx]
        X_dict[age_groups[ag]]['N'] = X[Ng_idx]
        X_dict[age_groups[ag]]['Ia'] = X[Iag_idx]
        X_dict[age_groups[ag]]['Ips'] = X[Ipsg_idx]
        X_dict[age_groups[ag]]['Ims'] = X[Imsg_idx]
        X_dict[age_groups[ag]]['Iss'] = X[Issg_idx]
        X_dict[age_groups[ag]]['Rq'] = X[Rqg_idx]
        X_dict[age_groups[ag]]['H'] = X[Hg_idx]
        X_dict[age_groups[ag]]['ICU'] = X[ICUg_idx]
        X_dict[age_groups[ag]]['D'] = X[Dg_idx]

    return X_dict

# Converts a dictionary into an X array
def dict_to_X(X_dict):
    X = np.zeros(num_compartments * num_age_groups, dtype=numpyArrayDatatype)

    for ag in range(num_age_groups):
        Sg_idx = ag*num_compartments + SEIR_groups.index('S_g')
        Eg_idx = ag*num_compartments + SEIR_groups.index('E_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Ng_idx = ag*num_compartments + SEIR_groups.index('N_g')
        Rg_idx = ag*num_compartments + SEIR_groups.index('R_g')
        Iag_idx = ag*num_compartments + SEIR_groups.index('Ia_g')
        Ipsg_idx = ag*num_compartments + SEIR_groups.index('Ips_g')
        Imsg_idx = ag*num_compartments + SEIR_groups.index('Ims_g')
        Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')

        X[Sg_idx] = X_dict[age_groups[ag]]['S']
        X[Eg_idx] = X_dict[age_groups[ag]]['E']
        X[Ig_idx] = X_dict[age_groups[ag]]['I']
        X[Rg_idx] = X_dict[age_groups[ag]]['R']
        X[Ng_idx] = X_dict[age_groups[ag]]['N']
        X[Iag_idx] = X_dict[age_groups[ag]]['Ia']
        X[Ipsg_idx] = X_dict[age_groups[ag]]['Ips']
        X[Imsg_idx] = X_dict[age_groups[ag]]['Ims']
        X[Issg_idx] = X_dict[age_groups[ag]]['Iss']
        X[Rqg_idx] = X_dict[age_groups[ag]]['Rq']
        X[Hg_idx] = X_dict[age_groups[ag]]['H']
        X[ICUg_idx] = X_dict[age_groups[ag]]['ICU']
        X[Dg_idx] = X_dict[age_groups[ag]]['D']

    return X



####################################
# Function that returns f(X(t), u(t))  = X(t+1). Does not change dynModel.
def get_F(dynModel, X, u, t):
    ''' Will return the next states given the state X and
    controls u. For this, it replaces the current state in
    dynModel for X, runs one step with controls u, extracts the
    resulting states, and re-starts the model to it's original
    state. Assumes as all functions above that X is ordered by
    compartment and then by group, and u by control and then by group.
    '''

    assert(X.shape == (num_compartments * num_age_groups, ))
    assert(u.shape == (num_controls * num_age_groups, ))

    # print("Started Get F")
    # print("*-*-*-*-*-*-*-*-*-*")

    # We save the initial time of dynModel to revert back to
    initial_time_of_model = dynModel.t
    initial_state_dict = dynModel.get_state(dynModel.t)


    # Covert array into dictionary
    X_dict = X_to_dict(X)

    # Determine the testing at time t given by u
    u_hat_dict, alphas = buildAlphaDict(u)


    m_tests = {}
    a_tests = {}
    for g in age_groups:
        m_tests[g] = u_hat_dict[g]['Nmtest_g']
        a_tests[g] = u_hat_dict[g]['Natest_g']

    B_H = {}
    B_ICU = {}

    #########################################
    # Below seems deprecated, because even if get_X_hat sequence does not use the bouncing variables,
    # it will still output controls B_H and B_ICU in new_u_hat_sequence.
    # To clean later.
    for g in age_groups:
        B_H[g] = u_hat_dict[g]['BounceH_g'] if (u_hat_dict[g]['BounceH_g'] != -1) else False
        # print("*****************************")
        # print("Bouncing from H for group {}: {}".format(g, B_H[g]))
        # print("Flow into H of group {}: {}".format(g, dynModel.groups[g].flow_H(initial_time_of_model)))
        B_ICU[g] = u_hat_dict[g]['BounceICU_g'] if (u_hat_dict[g]['BounceICU_g'] != -1) else False

    # Check if the bouncing strategy is prorrated
    prorrated = True
    for g in age_groups:
        if (B_H[g] is not False) or (B_ICU[g] is not False):
            prorrated = False

    # print("prorrated is", prorrated)
    ########################################

    # Write X_dict as current state of dynModel
    dynModel.write_state(t, X_dict)
    dynModel.t = t


    # Run a step of the dyn model
    if prorrated:
        dynModel.take_time_step(m_tests, a_tests, alphas, False, False)
    else:
        dynModel.take_time_step(m_tests, a_tests, alphas, B_H, B_ICU)
    #dynModel.take_time_step(m_tests, a_tests, alphas)

    # Get the current state
    # state_next_step = dynModel.get_state(dynModel.t)

    delta_x_next_step = dynModel.get_delta_X_over_delta_t(dynModel.t)

    # print("Time of Model after taking a step = {}".format(dynModel.t))

    # Make the dictionary into an array
    X_next_step = dict_to_X(delta_x_next_step)

    # Erase the states after t so as to reset the dyn model, also populate t with correct state
    dynModel.reset_time(initial_time_of_model)
    

    dynModel.write_state(dynModel.t, initial_state_dict)

    # print("Finished Running get_F")
    # print("*-*-*-*-*-*-*-*-*-*-*-*-*")

    assert dynModel.t == initial_time_of_model
    assert dynModel.get_state(dynModel.t) == initial_state_dict

    # Return dynModel to initial state
    return X_next_step


####################################
# our internal function here to calculate the contacts of a given age group with all other age groups
def calcContacts(dynModel, alphas, mixing_method, ag):
    # Calculate the multiplier
    
    # if t+dynModel.start_day < dynModel.parameters['days_before_gamma']:
    #     lock_state = "pre-gamma"
    #     prob_multiplier = dynModel.mixing_method["param_gamma_before"]
    # else:
    #     lock_state = "post-gamma"
    #     prob_multiplier = dynModel.mixing_method["param_gamma_after"]


    contacts_ag = np.zeros(num_age_groups, dtype=numpyArrayDatatype)
    for h in range(0, num_age_groups):
        contacts_ag[h] = n_contacts(dynModel.groups[age_groups[ag]], dynModel.groups[age_groups[h]], alphas, mixing_method)['total']

    
    return contacts_ag

####################################
# Calculate M, gamma and eta
def calculate_M_gamma_and_eta(dynModel):
    """Calculates the matrix M and the vectors gamma and eta that yield the objective"""

    # M should have number of rows equal to the len(mu(t))
    # and number of columns equal to the len of X(t)
    M = np.zeros((num_age_groups*num_controls, num_age_groups*num_compartments), dtype=numpyArrayDatatype)

    # Vector gamma should have len equal to the size of X(t)
    gamma = np.zeros(num_age_groups*num_compartments, dtype=numpyArrayDatatype)

    # Vector eta should have len equal to the size of X(t)
    eta = np.zeros(num_age_groups*num_compartments, dtype=numpyArrayDatatype)

    for ag in range(num_age_groups):
        # Get all the useful indices for the columns
        Sg_idx = ag*num_compartments + SEIR_groups.index('S_g')
        Eg_idx = ag*num_compartments + SEIR_groups.index('E_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Rg_idx = ag*num_compartments + SEIR_groups.index('R_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')

        LWorkg_idx = ag*num_controls + controls.index('work')
        LSchoolingg_idx = ag*num_controls + controls.index('school')
        LOtherg_idx = ag*num_controls + controls.index('other')
        LTransportg_idx = ag*num_controls + controls.index('transport')
        LLeisureg_idx = ag*num_controls + controls.index('leisure')

        # Rename parameters to make expressions similar to the Latex

        econ_activities = ["transport","leisure","other"]
        

        v_g = sum([dynModel.econ_params["employment_params"]["v"][age_groups[ag]][activity] for activity in econ_activities])


        nu = dynModel.econ_params["employment_params"]["nu"]

        small_eta = dynModel.econ_params["employment_params"]["eta"]

        school_value_g = dynModel.experiment_params['delta_schooling']*dynModel.econ_params['schooling_params'][age_groups[ag]]

        # number_of_groups = len(dynModel.groups)
        # assert number_of_groups == len(age_groups)

        small_gamma = dynModel.econ_params["employment_params"]["gamma"]

        v_g_schooling_ones = school_value_g
        v_g_employment_ones = v_g * (nu + small_eta + small_gamma)
        
    

        v_g_ones = v_g_schooling_ones + v_g_employment_ones

        vg_life_employment = dynModel.econ_params["econ_cost_death"][age_groups[ag]]
        xi = dynModel.experiment_params["xi"]
        # print(xi)

        # Matrix M should have only non-zero entries in the rows
        # corresponding to the lockdown decisions and the columns
        # corresponding to S_g E_g R_g and R^q_g
        # NOTICE THAT WE ASSUME HERE THAT R^q_g is in
        # no lockdown. If we want to implement the M with R^q
        # having the same lockdown we should have
        # (1-theta) * v_NLg  as well in the
        # column corresponding to R^q_g
        work_value_g = (v_g * nu)

        M[LWorkg_idx, Sg_idx] = work_value_g
        M[LWorkg_idx, Eg_idx] = work_value_g
        M[LWorkg_idx, Ig_idx] = work_value_g
        M[LWorkg_idx, Rg_idx] = work_value_g

        M[LSchoolingg_idx, Sg_idx] = school_value_g
        M[LSchoolingg_idx, Eg_idx] = school_value_g
        M[LSchoolingg_idx, Ig_idx] = school_value_g
        M[LSchoolingg_idx, Rg_idx] = school_value_g

        for ah in range(num_age_groups):
            LOtherh_idx = ah*num_controls + controls.index('other')
            LTransporth_idx = ah*num_controls + controls.index('transport')
            LLeisureh_idx = ah*num_controls + controls.index('leisure')

            activities_value_g = v_g * small_eta * (1/num_age_groups)
            M[LOtherh_idx, Sg_idx] = activities_value_g
            M[LOtherh_idx, Eg_idx] = activities_value_g
            M[LOtherh_idx, Ig_idx] = activities_value_g
            M[LOtherh_idx, Rg_idx] = activities_value_g

            M[LLeisureh_idx, Sg_idx] = activities_value_g
            M[LLeisureh_idx, Eg_idx] = activities_value_g
            M[LLeisureh_idx, Ig_idx] = activities_value_g
            M[LLeisureh_idx, Rg_idx] = activities_value_g

            M[LTransporth_idx, Sg_idx] = activities_value_g
            M[LTransporth_idx, Eg_idx] = activities_value_g
            M[LTransporth_idx, Ig_idx] = activities_value_g
            M[LTransporth_idx, Rg_idx] = activities_value_g




        # Vector gamma should have only nonzero elements in the
        # columns corresponding to states S_g E_g R_g and R^q
        # NOTICE THAT WE ASSUME HERE THAT R^q_g is not in
        # lockdown. If we want to implement the obj with R^q being
        # in lockdown, we should have v_NLg * theta in the
        # column corresponding to R^q_g
        baseline_economic_values = (
                        v_g * small_gamma
                        )
        gamma[Sg_idx] = baseline_economic_values
        gamma[Eg_idx] = baseline_economic_values
        gamma[Ig_idx] = baseline_economic_values
        gamma[Rg_idx] = baseline_economic_values

        gamma[Rqg_idx] = v_g_ones

        # Vector eta should have only nonzero elements in the
        # columns corresponding to D_g. We assume here that D(0)=0
        # but this should not change the decisions of the heur.
        eta[Dg_idx] = -(vg_life_employment + xi)

    return M, gamma, eta


########################################
# The constraints can be written as Gamma_x X(t) + Gamma_u u(t) <= K(t), for t = 0,...,T-1
# This function determines Gamma_x, Gamma_u and K(t). Gamma_x and Gamma_u are stored as matrices with
# one row for each constraint, compatible with calculate_all_constraint_coefs(.)
# K is stored as a dictionary with a key for every time t
# The boolean flag 'bounce_existing' determines whether to apply the more relaxed
# version of the capacity constraints, where existing patients can also be bounced
def calculate_all_constraints(dynModel, bounce_existing, h_cap_flag=False):
    """Calculate the matrices Gamma_x, Gamma_u and vectors K(t) that yield all the constraints"""

    # shorthand for a few useful parameters
    T = dynModel.time_steps
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups

    # calculate how many constraints there are in total (at a given time t)
    # H capacity: 1
    # BH : number of age groups
    # ICU capacity: 1
    # B ICU: number of age groups
    # M-tests : 1
    # A-tests: 1
    # lockdown: (number of age groups) x (number of activities)
    num_constraints = 3 + 2*num_age_groups

    # initialize Gamma_x and Gamma_u matrices with zeros
    Gamma_x = np.zeros((num_constraints, Xt_dim), dtype=numpyArrayDatatype)
    Gamma_u = np.zeros((num_constraints, ut_dim), dtype=numpyArrayDatatype)

    # right-hand-sides are time-varying; we store them in a matrix with one column for each time t
    K = np.zeros((num_constraints,T), dtype=numpyArrayDatatype)

    # index for current constraint
    curr_constr = 0

    # also store a string label for each constraint
    all_labels = []

    if h_cap_flag:
        ######## Constraint for H capacity
        for ag in range(0,num_age_groups):

            curr_group = dynModel.groups[age_groups[ag]]

            # Useful indices for elements of Gamma_x
            Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
            Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
            Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
            ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')

            # Useful indices for elements of Gamma_u
            BHg_idx = ag*num_controls + controls.index('BounceH_g')
            BICUg_idx = ag*num_controls + controls.index('BounceICU_g')

            # parameters
            mu_g = curr_group.parameters['mu']
            pICU_g = curr_group.parameters['p_ICU']
            pH_g = curr_group.parameters['p_H']
            lambda_H_R_g = curr_group.parameters['lambda_H_R']
            lambda_H_D_g = curr_group.parameters['lambda_H_D']

            Gamma_x[curr_constr,Ig_idx] = mu_g * pH_g
            Gamma_x[curr_constr,Issg_idx] = mu_g * (pH_g / (pH_g + pICU_g))
            Gamma_x[curr_constr,Hg_idx] = (1 - lambda_H_R_g - lambda_H_D_g)

            Gamma_u[curr_constr,BHg_idx] = -1

        # store right-hand-sides K(t) for every time t
        K[curr_constr,:] = dynModel.beds * dynModel.dt

        all_labels += ["H_capacity"]
        curr_constr += 1

    ################ Constraints for BH, for each age group
    for ag in range(num_age_groups):

        curr_group = dynModel.groups[age_groups[ag]]

        #Useful indices for the elements of Gamma_x
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')

        #Useful indices for the elements of Gamma_u
        BHg_idx = ag*num_controls + controls.index('BounceH_g')
        BICUg_idx = ag*num_controls + controls.index('BounceICU_g')

        # parameters
        mu_g = curr_group.parameters['mu']
        pICU_g = curr_group.parameters['p_ICU']
        pH_g = curr_group.parameters['p_H']

        Gamma_x[curr_constr,Ig_idx] = - mu_g * pH_g
        Gamma_x[curr_constr,Issg_idx] = - mu_g * (pH_g / (pH_g + pICU_g))
        Gamma_u[curr_constr,BHg_idx] = 1

        # check whether to allow bouncing existing patients
        if (bounce_existing):
            Gamma_x[curr_constr,Hg_idx] = -(1 - lambda_H_R_g - lambda_H_D_g)

        # right-hand-sides K(t) are all 0, so nothing to update
        all_labels += ["BH_%s" %age_groups[ag]]
        curr_constr += 1

    ######## Constraint for ICU capacity
    for ag in range(0,num_age_groups):

        curr_group = dynModel.groups[age_groups[ag]]

        #Useful indices for the elements of Gamma_x
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')

        #Useful indices for the elements of Gamma_u
        BHg_idx = ag*num_controls + controls.index('BounceH_g')
        BICUg_idx = ag*num_controls + controls.index('BounceICU_g')

        # parameters
        mu_g = curr_group.parameters['mu']
        pICU_g = curr_group.parameters['p_ICU']
        pH_g = curr_group.parameters['p_H']
        lambda_ICU_R_g = curr_group.parameters['lambda_ICU_R']
        lambda_ICU_D_g = curr_group.parameters['lambda_ICU_D']

        Gamma_x[curr_constr,Ig_idx] = mu_g * pICU_g
        Gamma_x[curr_constr,Issg_idx] = mu_g * (pICU_g / (pH_g + pICU_g))
        Gamma_x[curr_constr,ICUg_idx] = (1 - lambda_ICU_R_g - lambda_ICU_D_g)
        Gamma_u[curr_constr,BICUg_idx] = -1
     

    # store right-hand-sides K(t) for every time t
    K[curr_constr,:] = dynModel.icus * dynModel.dt
    # print(f"dynModel.parameters['global-parameters']['C_ICU']: {dynModel.parameters['global-parameters']['C_ICU']}")

    all_labels += ["ICU_capacity"]
    curr_constr += 1

    ################ Constraints for BICU, for each age group
    for ag in range(num_age_groups):

        curr_group = dynModel.groups[age_groups[ag]]

        #Useful indices for the elements of Gamma_x
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')

        #Useful indices for the elements of Gamma_u
        BHg_idx = ag*num_controls + controls.index('BounceH_g')
        BICUg_idx = ag*num_controls + controls.index('BounceICU_g')

        # Useful coefficients for Gamma_x and Gamma_u
        mu_g = curr_group.parameters['mu']
        pICU_g = curr_group.parameters['p_ICU']
        pH_g = curr_group.parameters['p_H']
        lambda_ICU_R_g = curr_group.parameters['lambda_ICU_R']
        lambda_ICU_D_g = curr_group.parameters['lambda_ICU_D']

        Gamma_x[curr_constr,Ig_idx] = - mu_g * pICU_g
        Gamma_x[curr_constr,Issg_idx] = - mu_g * (pICU_g / (pH_g + pICU_g))
        Gamma_u[curr_constr,BICUg_idx] = 1

        if (bounce_existing):
            Gamma_x[curr_constr,ICUg_idx] = -(1 - lambda_ICU_R_g - lambda_ICU_D_g)

        # right-hand-sides K(t) are all 0, so nothing to update
        all_labels += ["BICU_%s" %age_groups[ag]]
        curr_constr += 1

    ################ Constraint for M-test capacity
    # Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
    # Gamma_u[curr_constr,Nmtestg_idx_all] = 1
    # K[curr_constr,:] = dynModel.parameters['global-parameters']['C_mtest'] * dynModel.dt

    # all_labels += ["Mtest_cap"]
    # curr_constr += 1

    Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
    Gamma_u[curr_constr,Nmtestg_idx_all] = 1
    K[curr_constr,:] = dynModel.parameters['global-parameters']['C_mtest'] * dynModel.dt

    all_labels += ["Mtest_cap"]
    curr_constr += 1

    ################ Constraint for A-test capacity
    # Natestg_idx_all = slice(controls.index('Natest_g'),ut_dim,num_controls)
    # Gamma_u[curr_constr,Natestg_idx_all] = 1
    # K[curr_constr,:] = dynModel.parameters['global-parameters']['C_atest'] * dynModel.dt

    # all_labels += ["Atest_cap"]
    # curr_constr += 1

    Natestg_idx_all = slice(controls.index('Natest_g'),ut_dim,num_controls)
    Gamma_u[curr_constr,Natestg_idx_all] = 1
    K[curr_constr,:] =  dynModel.parameters['global-parameters']['C_atest'] * dynModel.dt

    all_labels += ["Atest_cap"]
    curr_constr += 1

    ################ Constraints for lockdown decisions, for each group and each activity
    # for ag in range(num_age_groups):
    #
    #     # loop over all activities
    #     for act in activities:
    #         Gamma_u[curr_constr,ag * num_controls + controls.index(act)] = 1
    #
    #         K[curr_constr,:] = 1  # right-hand-sides
    #
    #         all_labels += ["lockdown_%s_%s" %(age_groups[ag],act)]
    #         curr_constr += 1

    
    assert(len(all_labels) == num_constraints)


    return Gamma_x, Gamma_u, K, all_labels



####################################
# Calculate coefficients for each t period expression in the linearized objective
def calculate_objective_time_dependent_coefs(dynModel, k, xhat, uhat):
    """Calculates the coefficient vectors d_t, e_t that yield the linearized objective"""

    T = dynModel.time_steps
    M, gamma, eta = calculate_M_gamma_and_eta(dynModel)

    d = np.zeros((num_compartments * num_age_groups, T - k + 1), dtype=numpyArrayDatatype)
    e = np.zeros((num_controls * num_age_groups, T - k + 1), dtype=numpyArrayDatatype)

    for t in  range(k, T):
        d[:, t - k] = uhat[:, t - k] @ M + gamma
        e[:, t - k] = xhat[:, t - k] @ np.transpose(M)

    d[:, T - k] =  eta

    return d, e


########################################
# Consider a linear dynamical system of the form X(k+1)=Gamma_x(k) X(k) + Gamma_u(k) u(k) + K(k)
# and a linear expression of the form a*X(t)+b*u(t) for some a,b row vectors of suitable dimension.
# This function returns the coefficients for all the decisions u(k),...,u(T-1)
# appearing in all constraints and objective
# @profile
# @log_execution_time
def calculate_all_coefs(dynModel, k, Xhat_seq, uhat_seq, Gamma_x, Gamma_u, d_matrix, e_matrix, GPU=False):
    """Get coefficients for decisions appearing in a generic linear constraint in each period k,k+1,...
    Gamma_x and Gamma_u are matrices, for now. Can change to dictionaries later.
    Gamma_x: rows = number of "types" of constraints, columns = num_compartments * num_age_groups
    Gamma_u: rows = number of "types" of constraints, columns = num_controls * num_age_groups"""


    # shorthand for a few useful parameters
    T = dynModel.time_steps
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = Gamma_x.shape[0]

    assert( Xhat_seq.shape==(Xt_dim, T-k+1) )
    assert( uhat_seq.shape==(ut_dim, T-k) )
    assert(Gamma_x.shape ==(num_constraints,Xt_dim))
    assert(Gamma_u.shape ==(num_constraints,ut_dim))
    assert(d_matrix.shape ==(Xt_dim, T-k+1))
    assert(e_matrix.shape ==(ut_dim, T-k+1))

    # Some pre-processing:
    # Calculate matrices A and B, and vector c, at given Xhat_seq and uhat_seq, across all the necessary time indices
    # Hold these as dictionaries, where the key is the time t.
    
    At = np.zeros((T-k, Xt_dim, Xt_dim))
    Bt = np.zeros((T-k, Xt_dim, ut_dim))
    ct = np.zeros((T-k, Xt_dim))

    jacob_Xs = np.zeros((T-k, Xt_dim, Xt_dim))
    jacob_us = np.zeros((T-k, Xt_dim, ut_dim))

    #for t in range(k,T+1):
    for t in range(k,T):
        # print("calculate_all_coefs for loop, t = ", t)
        # get Xhat(t) and uhat(t)
        Xhat_t = Xhat_seq[:,t-k]
        uhat_t = uhat_seq[:,t-k]
        #print("uhat_t is", uhat_t)

        jacob_X = get_Jacobian_X_optimized(dynModel, Xhat_t, uhat_t, dynModel.mixing_method, t)
        jacob_u = get_Jacobian_u(dynModel, Xhat_t, uhat_t, dynModel.mixing_method, t)
 
        jacob_Xs[t-k] = jacob_X
        jacob_us[t-k] = jacob_u

    for t in range(k,T):
        jacob_X = jacob_Xs[t-k] 
        jacob_u = jacob_us[t-k]
        Xhat_t = Xhat_seq[:,t-k]
        uhat_t = uhat_seq[:,t-k]

        # Calculate linearization coefficients for X(t+1)
        At[t-k] = np.eye(Xt_dim) + dynModel.dt * jacob_X
        Bt[t-k] = dynModel.dt * jacob_u
        # print("Calculating C(t) for t = {}".format(t))
        # print("The time in the dynamical model is: {}".format(dynModel.t))
        # print("********************************")

        gf_vec = get_F(dynModel, Xhat_t, uhat_t, t)
        ct[t-k] = dynModel.dt * (gf_vec - jacob_X @ Xhat_t ) - jacob_u @ uhat_t #Revisit whether we need to multiply entire expression with dt

        # tol = 1e-6
        # mu_g = dynModel.groups[age_groups[0]].parameters['mu']
        # pICU_g = dynModel.groups[age_groups[0]].parameters['p_ICU']
        # pH_g = dynModel.groups[age_groups[0]].parameters['p_H']
        # lambda_ICU_R_g = dynModel.groups[age_groups[0]].parameters['lambda_ICU_R']
        # lambda_ICU_D_g = dynModel.groups[age_groups[0]].parameters['lambda_ICU_D']

        #print("1-lambda_ICU_R_g - lambda_ICU_R_g term = ", 1 - lambda_ICU_R_g - lambda_ICU_D_g)
        #print("mu_g p_ICU_g term = ", mu_g * pICU_g)
        #print("mu_g p_ICU_g / p_ICU_g + p_H_g term = ", mu_g * pICU_g / (pICU_g + pH_g))
        # print("calculation of At,Bt,ct for t = ", t)

        # x_approx = jacob_X @ Xhat_t
        # u_approx = jacob_u @ uhat_t
        for ag in range(num_age_groups):
            ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')
            # BounceICUg_idx = ag*num_controls + controls.index('BounceICU_g')
            # print("At[t][ICU] is equal to", At[t][ICUg_idx,:])
            # print("Bt[t][ICU] is equal to", Bt[t][ICUg_idx,:])
            # print("for group", ag, age_groups[ag], "ct[t][ICU] is equal to", ct[t][ICUg_idx])
            # print("components are", gf_vec[ICUg_idx], x_approx[ICUg_idx], u_approx[ICUg_idx])
            assert(abs(ct[t-k][ICUg_idx]) < 0.0001)

            # if (abs(ct[t][ICUg_idx]) > tol):
            #    print("ct[t][ICU] is not zero and equal to", ct[t][ICUg_idx])

    # All constraint coefficients are stored in dictionary u_constr_coeffs: u_constr_coeffs has a key for each
    # period t in {k,k+1,...,T-1}. The value for key t stores, in turn, another dictionary, which holds the constraint coefficients
    # of the constraints indexed with t.
    # In that dictionary, the key is the index of a constraint "type", and the value is a 2D numpy array with
    # (ut_dim) rows, and T-k columns (one for every time period k, k+1, ..., T-1). These are the coefficients for
    # all the controls u(k),...,u(T-1) appearing in the expression a*X(t) + b*u(t).
    # u_constr_coeffs = {}
    u_constr_coeffs = np.zeros((T-k, num_constraints, ut_dim, T-k), dtype=numpyArrayDatatype)

    # The linear expression for a constraint also has constants, which we store in a separate dictionary: constr_constants.
    # The constr_constants dictionary has a key for each period in {k,k+1,...,T-1}. The value for key t stores, in turn, another dictionary,
    # which holds the constants of the constraints indexed with t.
    # In that dictionary, the key is the index of a constraint "type", and the value is the constant corresponding to the specific
    # constraint type index and time period.
    # constr_constants = {}
    constr_constants = np.zeros((T-k, num_constraints), dtype=numpyArrayDatatype)

    # Initialize with zeros. (May want to try using sparse matrices here!)
    # for t in np.arange(k,T):
    #     u_constr_coeffs[t] = {}
    #     constr_constants[t] = {}
    #     for constr_index in range(num_constraints):
    #         u_constr_coeffs[t][constr_index] = np.zeros((ut_dim,T-k), dtype=numpyArrayDatatype)

    # All objective coefficients are stored in a 2D numpy array with (ut_dim) rows, and (T-k) columns
    # (one for every time period k, k+1, ..., T-1). Column with index t stores the coefficients in the objective for decision u_{k+t}, which
    # is of dimension (ut_dim). Note that for the objective we do not keep track of constant terms.
    u_obj_coeffs = np.zeros((ut_dim, T-k), dtype=numpyArrayDatatype)

    # We keep track of certain partial products of matrices / vectors that are useful
    # NOTE. When comparing this with Overleaf, note that we are only keeping track of
    # the relevant matrices for the current period t (i.e, ignoring t-1,t-2,etc.)
    # At_bar_dict = {}
    At_bar = np.zeros((T-k+1, Xt_dim, Xt_dim))
    Xt_bar = Xhat_seq[:,0]      # initialize with X(k)=Xhat(k)

    # print("Computing constants for all periods.")
    
    if GPU: 
        assert(torch.cuda.is_available())
        At_tensor = torch.from_numpy(At).cuda()
        Bt_tensor = torch.from_numpy(Bt).cuda()
        ct_tensor = torch.from_numpy(ct).cuda()
        At_bar_tensor = torch.from_numpy(At_bar).cuda()
        Gamma_x_tensor = torch.from_numpy(Gamma_x).cuda()
        Xt_bar_tensor = torch.from_numpy(Xt_bar).cuda()
        d_matrix_tensor = torch.from_numpy(d_matrix).cuda()

    for t in range(k,T): # loop over times k, k+1, ..., T - 1 to model constraints indexed with t
        # Calculate constants for period t
        for constr_index in range(num_constraints):
            if GPU:
                constr_constants[t-k][constr_index] = torch.matmul(Gamma_x_tensor[constr_index,:], Xt_bar_tensor).cpu().numpy()
            else:
                constr_constants[t-k][constr_index] = Gamma_x[constr_index,:] @ Xt_bar
            
        # print("Calculated constants for time {}".format(t))
        # Update auxiliary vector Xt_bar
        if GPU:
            Xt_bar_tensor = torch.matmul(At_tensor[t-k], Xt_bar_tensor) + ct_tensor[t-k]
        else:
            Xt_bar = At[t-k] @ Xt_bar + ct[t-k]


        # Calculate coefficients for all controls appearing in the constraint for period t
        # NOTE: The coefficients for control u(tau) are stored on column indexed (tau-k) of the 2D array

        for constr_index in range(num_constraints):
            # coefs for u[t]
            u_constr_coeffs[t-k][constr_index][:,t-k] = Gamma_u[constr_index,:]


        # Calculate coefficients for objective coefficient for u_t. Note that this is not the final coefficient of u_t.
        # Since the objective adds linear terms over all k, k+1..., T-1, u_t will receive additional contributions to its coefficient
        u_obj_coeffs[:,t-k] += e_matrix[:,t-k]

        # Initialize At_bar for tau=t-1
        if GPU:
            At_bar_tensor[t-k] = torch.eye(Xt_dim,Xt_dim).cuda()
        else:
            At_bar[t-k] = np.eye(Xt_dim,Xt_dim)

        for tau in range(t-1,k-1,-1):
            if GPU:
                At_bar_times_Bt_tensor = torch.matmul(At_bar_tensor[tau-k+1], Bt_tensor[tau-k])
                all_constraint_coefs_matrix = torch.matmul(Gamma_x_tensor, At_bar_times_Bt_tensor).cpu().numpy()
            else:
                At_bar_times_Bt = At_bar[tau-k+1] @ Bt[tau-k]
                all_constraint_coefs_matrix = Gamma_x @ At_bar_times_Bt

            for constr_index in range(num_constraints):
                # coefs for u[t-1], u[t-2], ..., u[k] in the constraints
                u_constr_coeffs[t-k][constr_index][:,tau-k] = all_constraint_coefs_matrix[constr_index, :]
            
            # coefs for u[t-1], u[t-2], ..., u[k] in the objective
            if GPU:
                u_obj_coeffs[:,tau-k] += torch.matmul(d_matrix_tensor[:,t-k], At_bar_times_Bt_tensor).cpu().numpy()
            else:
                u_obj_coeffs[:,tau-k] += d_matrix[:,t-k] @ At_bar_times_Bt

            # Update At_bar for next round
            if GPU:
                At_bar_tensor[tau-k] = torch.matmul(At_bar_tensor[tau-k+1], At_tensor[tau-k])
            else:
                At_bar[tau-k] = At_bar[tau-k+1] @ At[tau-k]

        # print("Computed constraint and obj coeff for time {}".format(t))

    # Now we handle the case of t=T
    if GPU:
        At_bar_tensor[T-k] = torch.eye(Xt_dim, Xt_dim).cuda()
    else:
        At_bar[T-k] = np.eye(Xt_dim,Xt_dim)

    # Add up the contribution of eta * X_T in the coefficients of decision u_t, t = k, ..., T-1
    for tau in range(T-1,k-1,-1):
        if GPU:
            # st1 = time()
            u_obj_coeffs[:,tau-k] += torch.matmul(torch.matmul(d_matrix_tensor[:,T-k], At_bar_tensor[tau-k+1]), Bt_tensor[tau-k]).cpu().numpy()
            At_bar_tensor[tau-k] = torch.matmul(At_bar_tensor[tau-k+1], At_tensor[tau-k])
            # en1 = time()
            # print("GPU: {}".format(int((en1 - st1) * 10**8) / 10**8))
        else:
            # st2 = time()
            u_obj_coeffs[:,tau-k] += d_matrix[:,T-k] @ At_bar[tau-k+1] @ Bt[tau-k]
            At_bar[tau-k] = At_bar[tau-k+1] @ At[tau-k]
            # en2 = time()
            # print("CPU: {}".format(int((en2 - st2) * 10**8) / 10**8))

    return u_constr_coeffs, constr_constants, u_obj_coeffs

def calculate_all_coefs_opt_cpu(dynModel, k, Xhat_seq, uhat_seq, Gamma_x, Gamma_u, d_matrix, e_matrix):
    """Get coefficients for decisions appearing in a generic linear constraint in each period k,k+1,...
    Gamma_x and Gamma_u are matrices, for now. Can change to dictionaries later.
    Gamma_x: rows = number of "types" of constraints, columns = num_compartments * num_age_groups
    Gamma_u: rows = number of "types" of constraints, columns = num_controls * num_age_groups"""

    # shorthand for a few useful parameters
    T = dynModel.time_steps
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = Gamma_x.shape[0]

    assert( Xhat_seq.shape==(Xt_dim, T-k+1) )
    assert( uhat_seq.shape==(ut_dim, T-k) )
    assert(Gamma_x.shape ==(num_constraints,Xt_dim))
    assert(Gamma_u.shape ==(num_constraints,ut_dim))
    assert(d_matrix.shape ==(Xt_dim, T-k+1))
    assert(e_matrix.shape ==(ut_dim, T-k+1))

    # Some pre-processing:
    # Calculate matrices A and B, and vector c, at given Xhat_seq and uhat_seq, across all the necessary time indices
    # Hold these as dictionaries, where the key is the time t.
    At = np.zeros((T-k, Xt_dim, Xt_dim))
    Bt = np.zeros((T-k, Xt_dim, ut_dim))
    ct = np.zeros((T-k, Xt_dim))
    jacob_Xs = np.zeros((T-k, Xt_dim, Xt_dim))
    jacob_us = np.zeros((T-k, Xt_dim, ut_dim))
    gf_vecs = np.zeros((T-k, Xt_dim))
    # start51 = time()
    #for t in range(k,T+1):
    for t in range(k,T):
        # print("calculate_all_coefs for loop, t = ", t)
        # get Xhat(t) and uhat(t)
        Xhat_t = Xhat_seq[:,t-k]
        uhat_t = uhat_seq[:,t-k]
        #print("uhat_t is", uhat_t)

        jacob_X = get_Jacobian_X_optimized(dynModel, Xhat_t, uhat_t, dynModel.mixing_method, t)
        jacob_u = get_Jacobian_u(dynModel, Xhat_t, uhat_t, dynModel.mixing_method, t)
 
        jacob_Xs[t-k] = jacob_X
        jacob_us[t-k] = jacob_u
        gf_vec = get_F(dynModel, Xhat_t, uhat_t, t)
        gf_vecs[t-k] = gf_vec

    ICUg_idx = list(range(SEIR_groups.index('ICU_g'), num_age_groups * num_compartments, num_compartments))

    At = np.tile(np.expand_dims(np.eye(Xt_dim), axis=0), (T-k, 1, 1)) + dynModel.dt * jacob_Xs
    Bt = dynModel.dt * jacob_us
    ct = dynModel.dt * (gf_vecs - np.squeeze(np.matmul(jacob_Xs, np.expand_dims(Xhat_seq.T[:-1], -1)), -1)) - np.squeeze(np.matmul(jacob_us, np.expand_dims(uhat_seq.T, -1)), -1)
    
    assert(abs(ct[:,ICUg_idx]) < 0.0001).all()

    # end51 = time()
    # print("5.1. [While]: Preprocessing {}".format(int((end51 - start51) * 10**5) / 10**5))

    # start52 = time()
    # All constraint coefficients are stored in dictionary u_constr_coeffs: u_constr_coeffs has a key for each
    # period t in {k,k+1,...,T-1}. The value for key t stores, in turn, another dictionary, which holds the constraint coefficients
    # of the constraints indexed with t.
    # In that dictionary, the key is the index of a constraint "type", and the value is a 2D numpy array with
    # (ut_dim) rows, and T-k columns (one for every time period k, k+1, ..., T-1). These are the coefficients for
    # all the controls u(k),...,u(T-1) appearing in the expression a*X(t) + b*u(t).
    # u_constr_coeffs = {}
    u_constr_coeffs = np.zeros((T-k, num_constraints, ut_dim, T-k), dtype=numpyArrayDatatype)

    # The linear expression for a constraint also has constants, which we store in a separate dictionary: constr_constants.
    # The constr_constants dictionary has a key for each period in {k,k+1,...,T-1}. The value for key t stores, in turn, another dictionary,
    # which holds the constants of the constraints indexed with t.
    # In that dictionary, the key is the index of a constraint "type", and the value is the constant corresponding to the specific
    # constraint type index and time period.
    # constr_constants = {}
    constr_constants = np.zeros((T-k, num_constraints), dtype=numpyArrayDatatype)

    # Initialize with zeros. (May want to try using sparse matrices here!)
    # for t in np.arange(k,T):
    #     u_constr_coeffs[t] = {}
    #     constr_constants[t] = {}
    #     for constr_index in range(num_constraints):
    #         u_constr_coeffs[t][constr_index] = np.zeros((ut_dim,T-k), dtype=numpyArrayDatatype)

    # All objective coefficients are stored in a 2D numpy array with (ut_dim) rows, and (T-k) columns
    # (one for every time period k, k+1, ..., T-1). Column with index t stores the coefficients in the objective for decision u_{k+t}, which
    # is of dimension (ut_dim). Note that for the objective we do not keep track of constant terms.
    u_obj_coeffs = np.zeros((ut_dim, T-k), dtype=numpyArrayDatatype)

    # We keep track of certain partial products of matrices / vectors that are useful
    # NOTE. When comparing this with Overleaf, note that we are only keeping track of
    # the relevant matrices for the current period t (i.e, ignoring t-1,t-2,etc.)
    # At_bar_dict = {}
    At_bar = np.zeros((T-k+1, Xt_dim, Xt_dim))
    Xt_bar = Xhat_seq[:,0]      # initialize with X(k)=Xhat(k)

    # print("Computing constants for all periods.")
    # end52 = time()
    # print("5.2. [While]: Initialization {}".format(int((end52 - start52) * 10**5) / 10**5))

    # start53 = time()
    for t in range(k,T): # loop over times k, k+1, ..., T - 1 to model constraints indexed with t
        # Calculate constants for period t
        constr_constants[t-k] = Gamma_x @ Xt_bar
            
        # Update auxiliary vector Xt_bar
        Xt_bar = At[t-k] @ Xt_bar + ct[t-k]

        # Calculate coefficients for all controls appearing in the constraint for period t
        # NOTE: The coefficients for control u(tau) are stored on column indexed (tau-k) of the 2D array

        u_constr_coeffs[t-k,:,:,t-k] = Gamma_u

        # Calculate coefficients for objective coefficient for u_t. Note that this is not the final coefficient of u_t.
        # Since the objective adds linear terms over all k, k+1..., T-1, u_t will receive additional contributions to its coefficient
        u_obj_coeffs[:,t-k] += e_matrix[:,t-k]

        # Initialize At_bar for tau=t-1
        At_bar[t-k] = np.eye(Xt_dim,Xt_dim)

        for tau in range(t-1,k-1,-1):
            At_bar_times_Bt = At_bar[tau-k+1] @ Bt[tau-k]
            all_constraint_coefs_matrix = Gamma_x @ At_bar_times_Bt

            u_constr_coeffs[t-k,:,:,tau-k] = all_constraint_coefs_matrix
            
            # coefs for u[t-1], u[t-2], ..., u[k] in the objective
            u_obj_coeffs[:,tau-k] += d_matrix[:,t-k] @ At_bar_times_Bt

            # Update At_bar for next round
            At_bar[tau-k] = At_bar[tau-k+1] @ At[tau-k]

    # print("Computed constraint and obj coeff for time {}".format(t))
    # end53 = time()
    # print("5.3. [While]: Compute recursive X and coefficients {}".format(int((end53 - start53) * 10**5) / 10**5))

    # start54 = time()
    # Now we handle the case of t=T

    At_bar[T-k] = np.eye(Xt_dim,Xt_dim)

    # # Add up the contribution of eta * X_T in the coefficients of decision u_t, t = k, ..., T-1
    for tau in range(T-1,k-1,-1):
        u_obj_coeffs[:,tau-k] += d_matrix[:,T-k] @ At_bar[tau-k+1] @ Bt[tau-k]
        At_bar[tau-k] = At_bar[tau-k+1] @ At[tau-k]
    # end54 = time()
    # print("5.4. [While]: Compute time step T {}".format(int((end54 - start54) * 10**5) / 10**5))

    return u_constr_coeffs, constr_constants, u_obj_coeffs


def calculate_all_coefs_opt_gpu(dynModel, k, Xhat_seq, uhat_seq, Gamma_x, Gamma_u, d_matrix, e_matrix):
    """Get coefficients for decisions appearing in a generic linear constraint in each period k,k+1,...
    Gamma_x and Gamma_u are matrices, for now. Can change to dictionaries later.
    Gamma_x: rows = number of "types" of constraints, columns = num_compartments * num_age_groups
    Gamma_u: rows = number of "types" of constraints, columns = num_controls * num_age_groups"""
    assert(torch.cuda.is_available())
    device = torch.device("cuda")

    # shorthand for a few useful parameters
    T = dynModel.time_steps
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = Gamma_x.shape[0]

    assert( Xhat_seq.shape==(Xt_dim, T-k+1) )
    assert( uhat_seq.shape==(ut_dim, T-k) )
    assert(Gamma_x.shape ==(num_constraints,Xt_dim))
    assert(Gamma_u.shape ==(num_constraints,ut_dim))
    assert(d_matrix.shape ==(Xt_dim, T-k+1))
    assert(e_matrix.shape ==(ut_dim, T-k+1))

    # Some pre-processing:
    # Calculate matrices A and B, and vector c, at given Xhat_seq and uhat_seq, across all the necessary time indices
    # Hold these as dictionaries, where the key is the time t.
    jacob_Xs = np.zeros((T-k, Xt_dim, Xt_dim))
    jacob_us = np.zeros((T-k, Xt_dim, ut_dim))
    gf_vecs = np.zeros((T-k, Xt_dim))
    
    # start51 = time()
    #for t in range(k,T+1):
    for t in range(k,T):
        # print("calculate_all_coefs for loop, t = ", t)
        # get Xhat(t) and uhat(t)
        Xhat_t = Xhat_seq[:,t-k]
        uhat_t = uhat_seq[:,t-k]
        #print("uhat_t is", uhat_t)

        jacob_X = get_Jacobian_X_optimized(dynModel, Xhat_t, uhat_t, dynModel.mixing_method, t)
        jacob_u = get_Jacobian_u(dynModel, Xhat_t, uhat_t, dynModel.mixing_method, t)
 
        jacob_Xs[t-k] = jacob_X
        jacob_us[t-k] = jacob_u
        gf_vec = get_F(dynModel, Xhat_t, uhat_t, t)
        gf_vecs[t-k] = gf_vec

    ICUg_idx = list(range(SEIR_groups.index('ICU_g'), num_age_groups * num_compartments, num_compartments))

    jacob_Xs_tensor = torch.from_numpy(jacob_Xs).cuda()
    jacob_us_tensor = torch.from_numpy(jacob_us).cuda()
    gf_vecs_tensor = torch.from_numpy(gf_vecs).cuda()
    Xhat_seq_tensor = torch.from_numpy(Xhat_seq[:, :-1]).cuda().T
    uhat_seq_tensor = torch.from_numpy(uhat_seq).cuda().T

    At = torch.eye(Xt_dim, device=device).unsqueeze(0).repeat(T-k, 1, 1) + dynModel.dt * jacob_Xs_tensor
    Bt = dynModel.dt * jacob_us_tensor
    ct = dynModel.dt * (gf_vecs_tensor - torch.matmul(jacob_Xs_tensor, Xhat_seq_tensor.unsqueeze(-1)).squeeze(-1)) - torch.matmul(jacob_us_tensor, uhat_seq_tensor.unsqueeze(-1)).squeeze(-1)

    assert(abs(ct[:,ICUg_idx]) < 0.0001).all()

    # end51 = time()
    # print("5.1. [While]: Preprocessing {}".format(int((end51 - start51) * 10**5) / 10**5))

    # start52 = time()
    # All constraint coefficients are stored in dictionary u_constr_coeffs: u_constr_coeffs has a key for each
    # period t in {k,k+1,...,T-1}. The value for key t stores, in turn, another dictionary, which holds the constraint coefficients
    # of the constraints indexed with t.
    # In that dictionary, the key is the index of a constraint "type", and the value is a 2D numpy array with
    # (ut_dim) rows, and T-k columns (one for every time period k, k+1, ..., T-1). These are the coefficients for
    # all the controls u(k),...,u(T-1) appearing in the expression a*X(t) + b*u(t).
    # u_constr_coeffs = {}
    u_constr_coeffs = np.zeros((T-k, num_constraints, ut_dim, T-k), dtype=numpyArrayDatatype)
    u_constr_coeffs_star = np.zeros((T-k, num_constraints, ut_dim, T-k), dtype=numpyArrayDatatype)

    # The linear expression for a constraint also has constants, which we store in a separate dictionary: constr_constants.
    # The constr_constants dictionary has a key for each period in {k,k+1,...,T-1}. The value for key t stores, in turn, another dictionary,
    # which holds the constants of the constraints indexed with t.
    # In that dictionary, the key is the index of a constraint "type", and the value is the constant corresponding to the specific
    # constraint type index and time period.

    # Initialize with zeros. (May want to try using sparse matrices here!)
    # for t in np.arange(k,T):
    #     u_constr_coeffs[t] = {}
    #     constr_constants[t] = {}
    #     for constr_index in range(num_constraints):
    #         u_constr_coeffs[t][constr_index] = np.zeros((ut_dim,T-k), dtype=numpyArrayDatatype)

    # All objective coefficients are stored in a 2D numpy array with (ut_dim) rows, and (T-k) columns
    # (one for every time period k, k+1, ..., T-1). Column with index t stores the coefficients in the objective for decision u_{k+t}, which
    # is of dimension (ut_dim). Note that for the objective we do not keep track of constant terms.
    u_obj_coeffs_star = np.zeros((ut_dim, T-k), dtype=numpyArrayDatatype)

    # We keep track of certain partial products of matrices / vectors that are useful
    # NOTE. When comparing this with Overleaf, note that we are only keeping track of
    # the relevant matrices for the current period t (i.e, ignoring t-1,t-2,etc.)
    # At_bar_dict = {}
    At_star = torch.cuda.DoubleTensor(T-k+1, T-k+1, Xt_dim, Xt_dim).fill_(0)
    Xt_star = torch.cuda.DoubleTensor(T-k, Xt_dim).fill_(0)
    Xt_star[0] = torch.from_numpy(Xhat_seq[:,0]).cuda()
    Gamma_x_tensor = torch.from_numpy(Gamma_x).cuda()
    d_matrix_tensor = torch.from_numpy(d_matrix).cuda()

    # print("Computing constants for all periods.")
    # end52 = time()
    # print("5.2. [While]: Initialization {}".format(int((end52 - start52) * 10**5) / 10**5))

    # start53 = time()
    # Compute At_bar before everything else
    for t in range(k, T+1):
        # Initialize At_bar for tau=t-1
        At_star[t-k, t-k] = torch.eye(Xt_dim, device=device)
    for t in range(T-1, k-1, -1):
        At_star[t-k+1:T-k+1, t-k] = torch.tensordot(At_star[t-k+1:T-k+1, t-k+1], At[t-k], dims=([-1], [0]))

    # # Compute Xt_bar
    for t in range(k, T-1):
        Xt_star[t-k+1] = torch.matmul(At[t-k], Xt_star[t-k]) + ct[t-k]

    constr_constants_star = torch.matmul(Gamma_x_tensor, Xt_star.T).T.cpu().numpy()
    for t in range(k, T):
        u_constr_coeffs_star[t-k,:,:,t-k] = Gamma_u
        u_obj_coeffs_star[:,t-k] += e_matrix[:,t-k]
        # Replaced tau for loop
        if t-k > 0:
            At_star_times_Bt = torch.matmul(At_star[t-k, 1:t-k+1], Bt[:t-k]).permute(1,2,0)
            all_constraint_coefs_matrix_star = torch.tensordot(Gamma_x_tensor, At_star_times_Bt, dims=([-1], [0]))
            u_constr_coeffs_star[t-k,:,:,:t-k] = all_constraint_coefs_matrix_star.cpu().numpy()
            u_obj_coeffs_star[:, :t-k] += torch.tensordot(d_matrix_tensor[:,t-k], At_star_times_Bt, dims=([-1], [0])).cpu().numpy()
    At_star_times_Bt = torch.matmul(At_star[T-k, 1:T-k+1], Bt[:T-k]).permute(1,2,0)
    u_obj_coeffs_star[:,:T-k] += torch.tensordot(d_matrix_tensor[:,T-k], At_star_times_Bt, dims=([-1], [0])).cpu().numpy()

    return u_constr_coeffs_star, constr_constants_star, u_obj_coeffs_star



####################################
# Takes the dynModel, runs the DynModel with the uhat_seq
# and returns the final reward. Restores the dynModel to its
# original state.
def get_real_reward(dynModel, uhat_seq):
    
    # some boolean flags for running the heuristic
    use_bounce_var = True   # whether to use the optimal bounce variables when forecasting the new X_hat
    bounce_existing = False   # whether to allow bouncing existing patients

    original_time = dynModel.t

    # if not debugging:
    print(max(uhat_seq.shape[1]-dynModel.END_DAYS,0))
    for i in range(uhat_seq.shape[1]):
    
        u_dict, alpha_dict = buildAlphaDict(uhat_seq[:,i])

        m_tests = {}
        a_tests = {}
        BH = {}
        BICU = {}
        for ag in age_groups:
            BH[ag] = u_dict[ag]['BounceH_g']
            BICU[ag] = u_dict[ag]['BounceICU_g']
            m_tests[ag] = u_dict[ag]['Nmtest_g']
            a_tests[ag] = u_dict[ag]['Natest_g']

        # take one time step in dynamical system
        if(use_bounce_var):
            dynModel.take_time_step(m_tests, a_tests, alpha_dict, BH, BICU)
        else:
            dynModel.take_time_step(m_tests, a_tests, alpha_dict)

    reward = dynModel.get_total_reward()

    dynModel.reset_time(original_time)
    
    return reward




####################################
# Main function: runs the linearization heuristic
# @profile
# @log_execution_time
def run_heuristic_linearization(dynModel, trust_region_radius=0.2, max_inner_iterations_mult=2, initial_uhat="dynamic_gradient", optimize_bouncing=True):
    """Run the heuristic based on linearization. Takes a dynamical model, resets the time to 0, and runs it following the linearization heuristic. Returns the dynamical model after running it."""

    # age_groups = dynModel.groups.keys()
    # num_age_groups = len(age_groups)
    # num_compartments = len(SEIR_groups)
    # num_controls = len(controls)
    # num_activities = len(activities)

    # Xt_dim = num_compartments * num_age_groups
    # ut_dim = num_controls * num_age_groups


    # start1 = time()
    dynModel.reset_time(0)

    # trust_region_radius = 0.1
    threshold = 0.001
    max_inner_iterations = max_inner_iterations_mult / trust_region_radius if trust_region_radius>0 else 1

    # shorthand for a few useful parameters
    T = dynModel.time_steps
    
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = 3 + 2*num_age_groups

    # some boolean flags for running the heuristic
    use_bounce_var = True   # whether to use the optimal bounce variables when forecasting the new X_hat
    bounce_existing = False   # whether to allow bouncing existing patients

    test_freq = 1
    lockdown_freq = 1

    if 'test_freq' in dynModel.experiment_params:
        test_freq = dynModel.experiment_params['test_freq']

    if 'lockdown_freq' in dynModel.experiment_params:
        lockdown_freq = dynModel.experiment_params['lockdown_freq']

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
   
    # optimal decisions
    # uopt_seq = np.zeros((ut_dim,T), dtype=numpyArrayDatatype)

    # initialize a starting u_hat sequence with full lockdown, 
    # no tests, and no bouncing
    uhat_seq = np.zeros((ut_dim,T), dtype=numpyArrayDatatype)

    # for now, homogenous testing
    Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
    uhat_seq[Nmtestg_idx_all,:] = dynModel.parameters['global-parameters']['C_mtest']/num_age_groups

    Natestg_idx_all = slice(controls.index('Natest_g'), ut_dim,num_controls)
    uhat_seq[Natestg_idx_all,:] = dynModel.parameters['global-parameters']['C_atest']/num_age_groups

    # Initialize lockdown policy for first u_hat

    if initial_uhat in ["dynamic_gradient", "time_gradient"]:
        h = initial_uhat
        n = "xi-%d_icus-%d_testing-%s_natests-%d_nmtests-%d_T-%d_startday-%d_groups-%s_dschool-%f_eta-%f_freq-%d-%d.yaml"%(
            dynModel.experiment_params["xi"]*0.1,
            dynModel.experiment_params["icus"]*10000,
            "homogeneous",
            dynModel.parameters['global-parameters']['C_atest']*10000,
            dynModel.parameters['global-parameters']['C_mtest']*10000,
            dynModel.time_steps - dynModel.END_DAYS,
            dynModel.start_day,
            "all",
            dynModel.experiment_params["delta_schooling"],
            dynModel.econ_params["employment_params"]["eta"],
            90,
            14
        )

        with open("benchmarks/results/%s/%s"%(h,n)) as file:
        # with open("scripts/results/experiments/%s/%s"%(h,n)) as file:
            result = yaml.load(file, Loader=yaml.FullLoader)

        # Starting the uhat_seq 
        for t in range(T):
            for ag in range(num_age_groups):
                for act in activities:
                    act_index = (controls.index(act) + num_controls * ag)
                    uhat_seq[act_index,t] = result["policy"][t][age_groups[ag]][act]
    
    if initial_uhat == "random":
        for act in activities:
            act_indices = slice(controls.index(act), ut_dim, num_controls)
            # Random lockdowns
            uhat_seq[act_indices,:] = random.uniform(0,1)
        # print("Uhat {}".format(uhat_seq))

    if initial_uhat == "full_lockdown":
        # the controls are initialized in full lockdown
        pass 

    if initial_uhat == "full_open":
        for act in activities:
            act_indices = slice(controls.index(act), ut_dim, num_controls)
            # Random lockdowns
            uhat_seq[act_indices,:] = 1


    # and home lockdown variables all 1
    lock_home_idx_all = slice(controls.index('home'),ut_dim,num_controls)
    uhat_seq[lock_home_idx_all,:] = 1.0

    # a python list with the indices for all home lockdown decisions for all groups and periods

    # end1 = time()
    # if debugging:
        # print("1. [Outside]: Initialization takes {} seconds".format(int((end1 - start1) * 10**5) / 10**5))

    dynModel.shadowPrices = {}

    for k in range(T):

        # start2 = time()
        lock_home_idx_all_times = [controls.index('home') + i*num_controls for i in range((T-k)*num_age_groups)]

        lock_work_idx_all_times = [controls.index('work') + i*num_controls for i in range((T-k)*num_age_groups)]

        lock_other_idx_all_times = [controls.index('other') + i*num_controls for i in range((T-k)*num_age_groups)]

        lock_school_idx_all_times = [controls.index('school') + i*num_controls for i in range((T-k)*num_age_groups)]

        lock_leisure_idx_all_times = [controls.index('leisure') + i*num_controls for i in range((T-k)*num_age_groups)]

        lock_transport_idx_all_times = [controls.index('transport') + i*num_controls for i in range((T-k)*num_age_groups)]

        all_lockdowns_idx_all_times =  lock_work_idx_all_times + lock_other_idx_all_times + lock_school_idx_all_times + lock_leisure_idx_all_times + lock_transport_idx_all_times

        a_test_idx_all_times = [controls.index('Natest_g') + i*num_controls for i in range((T-k)*num_age_groups)]

        m_test_idx_all_times = [controls.index('Nmtest_g') + i*num_controls for i in range((T-k)*num_age_groups)]
        
        icu_bouncing_idx_all_times =  [controls.index('BounceICU_g') + i*num_controls for i in range((T-k)*num_age_groups)]

        hospital_bouncing_idx_all_times =  [controls.index('BounceH_g') + i*num_controls for i in range((T-k)*num_age_groups)]

        all_bouncing_idx_all_times = hospital_bouncing_idx_all_times + icu_bouncing_idx_all_times

        inner_iterations = 0
        u_hat_lockdown_difference = threshold + 1

        reward = get_real_reward(dynModel, uhat_seq)

        all_u_hat_seq_and_rewards = [(uhat_seq, reward)]

        best_uhat_and_reward = (uhat_seq, reward)

        # end2 = time()
        # print("2. [For]: For init takes {}".format(int((end2 - start2) * 10**5) / 10**5))

        print(f"Total Reward at the start of the inner iterations: {reward}")

        while inner_iterations < max_inner_iterations and u_hat_lockdown_difference > threshold:

            # start_it = time()
            # print("\n\n HEURISTIC RUNNING FOR TIME k= {}.".format(k))

            # start3 = time()
            # calculate state trajectory X_hat and corresponging controls new_uhat
            Xhat_seq, uhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq, use_bounce_var)

            assert( np.shape(Xhat_seq) == (Xt_dim,T-k+1) )
            assert( np.shape(uhat_seq) == (ut_dim,T-k) )

            # end3 = time()
            # print("3. [While]: Compute trajectory takes {}".format(int((end3 - start3) * 10**5) / 10**5))

            ICUidx_all = slice(SEIR_groups.index('ICU_g'), Xt_dim, num_compartments)

            # start4 = time()

            # calculate objective parameters d, e
            D,E = calculate_objective_time_dependent_coefs(dynModel, k, Xhat_seq, uhat_seq)

            # end4 = time()
            # print("4. [While]: Compute objective parameters d,e takes {}".format(int((end4 - start4) * 10**5) / 10**5))

            # start5 = time()

            # get coefficients for decisions in all constraints and objective
            if use_gpu:
                constr_coefs, constr_consts, obj_coefs = calculate_all_coefs_opt_gpu(dynModel,k,Xhat_seq,uhat_seq,Gamma_x,Gamma_u,D,E)
            else:
                constr_coefs, constr_consts, obj_coefs = calculate_all_coefs_opt_cpu(dynModel,k,Xhat_seq,uhat_seq,Gamma_x,Gamma_u,D,E)
            
            # end5 = time()
            # print("5. [While]: Compute coefficients and constraints takes {}".format(int((end5 - start5) * 10**5) / 10**5))

            assert( np.shape(obj_coefs) == (ut_dim,T-k) )
            assert( len(constr_coefs) == T-k )
            assert( len(constr_consts) == T-k )
            for t in range(k, T):
                assert( len(constr_coefs[t-k]) == num_constraints )
                assert( len(constr_consts[t-k]) == num_constraints )

                for i in range(num_constraints):
                    assert( np.shape(constr_coefs[t-k][i])==np.shape(uhat_seq) )
                    assert( np.shape(constr_consts[t-k][i])==() )

            # start6 = time()
            # start61 = time()
            # create empty model
            mod = gb.Model("Linearization Heuristic")

            mod.setParam( 'OutputFlag', False )     # make Gurobi silent
            mod.setParam( 'LogFile', "" )

            # mod.setParam('Threads', 1)
            mod.Params.threads = __gurobi_threads

            mod.Params.DualReductions = 0  # change this to get explicit infeasible or unbounded

            # add all decisions using matrix format, and also specify objective coefficients
            obj_vec = np.reshape(obj_coefs, (ut_dim*(T-k),), 'F')  # reshape by reading along rows first
            # obj_vec = np.zeros(ut_dim* (T-k))

            upper_bounds = np.ones(np.shape(obj_vec), dtype=numpyArrayDatatype) * np.inf
            lower_bounds = np.zeros(np.shape(obj_vec), dtype=numpyArrayDatatype)

            # Optimized
            np_all_lockdowns_idx_all_times = np.array(all_lockdowns_idx_all_times)
            np_lock_home_idx_all_times = np.array(lock_home_idx_all_times)
            np_a_test_idx_all_times = np.array(a_test_idx_all_times)
            np_m_test_idx_all_times = np.array(m_test_idx_all_times)

            # Work, Other, School, Leisure, Transport
            lower_bounds[np_all_lockdowns_idx_all_times] = np.maximum(uhat_seq[np_all_lockdowns_idx_all_times%ut_dim, np_all_lockdowns_idx_all_times//ut_dim] - trust_region_radius, 0)
            upper_bounds[np_all_lockdowns_idx_all_times] = np.minimum(uhat_seq[np_all_lockdowns_idx_all_times%ut_dim, np_all_lockdowns_idx_all_times//ut_dim] + trust_region_radius, 1)

            # Home
            lower_bounds[np_lock_home_idx_all_times] = 1
            upper_bounds[np_lock_home_idx_all_times] = 1

            # A tests
            lower_bounds[np_a_test_idx_all_times] = np.maximum(uhat_seq[np_a_test_idx_all_times%ut_dim, np_a_test_idx_all_times//ut_dim] - (trust_region_radius * dynModel.parameters['global-parameters']['C_atest']), 0)
            upper_bounds[np_a_test_idx_all_times] = np.minimum(uhat_seq[np_a_test_idx_all_times%ut_dim, np_a_test_idx_all_times//ut_dim] + (trust_region_radius * dynModel.parameters['global-parameters']['C_atest']), dynModel.parameters['global-parameters']['C_atest'])

            # M tests
            lower_bounds[np_m_test_idx_all_times] = np.maximum(uhat_seq[np_m_test_idx_all_times%ut_dim, np_m_test_idx_all_times//ut_dim] - (trust_region_radius * dynModel.parameters['global-parameters']['C_mtest']), 0)
            upper_bounds[np_m_test_idx_all_times] = np.minimum(uhat_seq[np_m_test_idx_all_times%ut_dim, np_m_test_idx_all_times//ut_dim] + (trust_region_radius * dynModel.parameters['global-parameters']['C_mtest']), dynModel.parameters['global-parameters']['C_mtest'])

            # for i in range(len(obj_vec)):
            #     if i in all_lockdowns_idx_all_times:
            #         lower_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim] - trust_region_radius, 0)
                    
            #         assert lower_bounds[i] >= 0

            #         upper_bounds[i] = min(uhat_seq[i%ut_dim, i//ut_dim] + trust_region_radius, 1)

            #     if i in lock_home_idx_all_times:
            #         upper_bounds[i] = 1
            #         lower_bounds[i] = 1

            #     if i in a_test_idx_all_times:
            #         lower_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim] - (trust_region_radius * dynModel.parameters['global-parameters']['C_atest']), 0)
                    
            #         assert lower_bounds[i] >= 0

            #         upper_bounds[i] = min(uhat_seq[i%ut_dim, i//ut_dim] + (trust_region_radius * dynModel.parameters['global-parameters']['C_atest']), dynModel.parameters['global-parameters']['C_atest'])

            #     if i in m_test_idx_all_times:
            #         lower_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim] - (trust_region_radius * dynModel.parameters['global-parameters']['C_mtest']), 0)
                    
            #         assert lower_bounds[i] >= 0

            #         upper_bounds[i] = min(uhat_seq[i%ut_dim, i//ut_dim] + (trust_region_radius * dynModel.parameters['global-parameters']['C_mtest']), dynModel.parameters['global-parameters']['C_mtest'])


                # if i in lock_work_idx_all_times:
                #     lower_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim] - trust_region_radius, 0)
                #     upper_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim]  trust_region_radius, 1)
                # if i in lock_other_idx_all_times:
                #     lower_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim] - trust_region_radius, 0)
                #     upper_bounds[i] = 0
                # if i in lock_school_idx_all_times:
                #     lower_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim] - trust_region_radius, 0)
                #     upper_bounds[i] = 0
                # if i in lock_leisure_idx_all_times:
                #     lower_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim] - trust_region_radius, 0)
                #     upper_bounds[i] = 0
                # if i in lock_transport_idx_all_times:
                #     lower_bounds[i] = max(uhat_seq[i%ut_dim, i//ut_dim] - trust_region_radius, 0)
                #     upper_bounds[i] = 0

            u_vars_vec = mod.addMVar( np.shape(obj_vec), lb=lower_bounds, ub=upper_bounds, obj=obj_vec, name="u")

            # mod.addConstrs(u_vars_vec[i] >= lower_bounds[i] for i in range(ut_dim * (T-k)))

            # mod.addConstrs(u_vars_vec[i] <= upper_bounds[i] for i in range(ut_dim * (T-k)))

            # mod.addConstrs(u_vars_vec[i] <= upper_bounds[i] for i in range(ut_dim * (T-k)))

            # Sense -1 indicates a maximization problem
            mod.ModelSense = -1

            # mod.addConstrs((u_vars_vec[i]==1 for i in lock_home_idx_all_times if i < len(obj_vec)), name=("home_lock"))
            #
            # mod.addConstrs((u_vars_vec[i]>=0.24 for i in lock_work_idx_all_times if i < len(obj_vec)), name=("work_lock_lb"))


            # No Constraint on Transport
            # work_index = controls.index('work')
            # transport_index = controls.index('transport')
            # TransportConstMatrix = np.zeros(((T-k) * num_age_groups, ut_dim * (T-k)), dtype=numpyArrayDatatype)
            # TransportConstRHSVector = np.zeros(((T-k) *num_age_groups,), dtype=numpyArrayDatatype)
            
            # work_index = controls.index('work')
            # transport_index = controls.index('transport')
            # TransportConstMatrix = np.zeros(((T-k) * num_age_groups, ut_dim * (T-k)), dtype=numpyArrayDatatype)
            # TransportConstRHSVector = np.zeros(((T-k) *num_age_groups,), dtype=numpyArrayDatatype)

            # for i in range((T-k) * num_age_groups):
            #     work_idx = work_index + i*num_controls
            #     transport_idx = transport_index + i*num_controls

            #     TransportConstMatrix[i, work_idx] = -dynModel.transport_lb_work_fraction
            #     TransportConstMatrix[i, transport_idx] = 1


            # mod.addMConstrs(TransportConstMatrix, u_vars_vec, ">", TransportConstRHSVector, name=("transport_lock_lb"))

            # end61 = time()
            # print("6.1. [While]: Create vars {}".format(int((end61 - start61) * 10**5) / 10**5))

            # start62 = time()
            # Weekly testing constraints
            if test_freq > 1 and (dynModel.parameters['global-parameters']['C_atest'] > 0 or dynModel.parameters['global-parameters']['C_mtest'] > 0):
                m_test_id = controls.index('Nmtest_g')
                a_test_id = controls.index('Natest_g')

                # m_test_idx_range = list(range(m_test_id, num_age_groups * num_controls, num_controls))
                # a_test_idx_range = list(range(a_test_id, num_age_groups * num_controls, num_controls))
                
                # if (k % lockdown_freq) != 0:
                #     mod.addConstr(u_vars_vec[m_test_idx_range] == dynModel.m_tests_controls[k-1][age_groups[:]])
                #     mod.addConstr(u_vars_vec[a_test_idx_range] == dynModel.a_tests_controls[k-1][age_groups[:]])

                m_test_idx_lhs = []
                m_test_idx_rhs = []

                a_test_idx_lhs = []
                a_test_idx_rhs = []

                for time_index in range(T - dynModel.END_DAYS):

                    if time_index == k and (time_index % lockdown_freq) != 0:
                        #Fix the first control to be equal to the control at time k-1
                        for ag in range(num_age_groups):
                            m_test_idx = m_test_id + ag * num_controls
                            a_test_idx = a_test_id + ag * num_controls

                            mod.addConstr(u_vars_vec[m_test_idx] == dynModel.m_tests_controls[k-1][age_groups[ag]])
                            mod.addConstr(u_vars_vec[a_test_idx] == dynModel.a_tests_controls[k-1][age_groups[ag]])

                    
                    if time_index > k and (time_index % lockdown_freq) != 0:
                        for ag in range(num_age_groups):
                            m_test_idx = m_test_id + ag * num_controls
                            a_test_idx = a_test_id + ag * num_controls
                            
                            m_test_idx_lhs.append((time_index - k) * ut_dim + m_test_idx)
                            m_test_idx_rhs.append((time_index - k - 1) * ut_dim + m_test_idx)

                            a_test_idx_lhs.append((time_index - k) * ut_dim + a_test_idx)
                            a_test_idx_rhs.append((time_index - k - 1) * ut_dim + a_test_idx)

                            # mod.addConstr(u_vars_vec[(time_index - k) * ut_dim + m_test_idx] == u_vars_vec[(time_index - k - 1) * ut_dim + m_test_idx])
                            # mod.addConstr(u_vars_vec[(time_index - k) * ut_dim + a_test_idx] == u_vars_vec[(time_index - k - 1) * ut_dim + a_test_idx])

                mod.addConstr(u_vars_vec[m_test_idx_lhs] == u_vars_vec[m_test_idx_rhs])
                mod.addConstr(u_vars_vec[a_test_idx_lhs] == u_vars_vec[a_test_idx_rhs])


            # Biweekly lockdown constraints
            if lockdown_freq > 1:

                act_test_idx_lhs = []
                act_test_idx_rhs = []

                for time_index in range(T - dynModel.END_DAYS):

                    if time_index == k and (time_index % lockdown_freq) != 0:
                        #Fix the first control to be equal to the control at time k-1
                        for ag in range(num_age_groups):
                            for act in activities:
                                act_lock_id = controls.index(act)
                                act_lock_idx = act_lock_id + ag * num_controls

                                mod.addConstr(u_vars_vec[act_lock_idx] == dynModel.lockdown_controls[k-1][age_groups[ag]][act], name="biweekly-initial")
                    
                    if time_index > k and (time_index % lockdown_freq) != 0:
                        for ag in range(num_age_groups):
                            for act in activities:
                                act_lock_id = controls.index(act)
                                act_lock_idx = act_lock_id + ag * num_controls

                                act_test_idx_lhs.append((time_index - k) * ut_dim + act_lock_idx)
                                act_test_idx_rhs.append((time_index - k - 1) * ut_dim + act_lock_idx)
                                # mod.addConstr(u_vars_vec[(time_index - k) * ut_dim + act_lock_idx] == u_vars_vec[(time_index - k - 1) * ut_dim + act_lock_idx])
                mod.addConstr(u_vars_vec[act_test_idx_lhs] == u_vars_vec[act_test_idx_rhs])

            # end62 = time()
            # print("6.2. [While]: Weekly constraints {}".format(int((end62 - start62) * 10**5) / 10**5))

            # start63 = time()
            ConstMatrix = np.zeros(((T-k) * num_constraints, ut_dim * (T-k)), dtype=numpyArrayDatatype)

            ConstRHS = np.zeros(((T-k) * num_constraints,), dtype=numpyArrayDatatype)
            for t in range(k,T):
                #print("Time %d number of constraints %d" %(t,len(constr_coefs[t])))
                for con in range(num_constraints):
                    cons_vec = np.reshape(constr_coefs[t-k][con], (ut_dim * (T-k),), 'F').transpose()
                    ConstMatrix[(t-k) * num_constraints + con, :] = cons_vec
                    ConstRHS[(t-k) * num_constraints + con] = K[con,t] - constr_consts[t-k][con]
                    # cname = ("%s[t=%d]" %(all_labels[con],t))
                    # expr = (u_vars_vec @ cons_vec) + (constr_consts[t][con])
                    # mod.addConstr((u_vars_vec @ cons_vec) + (constr_consts[t][con]) <= K[con,t], name=cname)
            
            all_const = mod.addMConstrs(ConstMatrix, u_vars_vec, "<", ConstRHS)

            # optimize the model

            # ToDo: Remove when running as it reduces efficiency
            # mod.Params.InfUnbdInfo = 1

            # eye = np.eye(u_vars_vec.shape[0])
            # zeros = np.zeros(u_vars_vec.shape[0])
            # lower_constrs = mod.addMConstrs(eye, u_vars_vec, ">=", zeros)

            mod.addConstr(u_vars_vec >= lower_bounds)
            mod.addConstr(u_vars_vec <= upper_bounds)

            # mod.addConstrs(u_vars_vec[i] >= 0 for i in range(ut_dim * (T-k)))
            # mod.addConstrs(u_vars_vec[i] >= 0 for i in range(ut_dim * (T-k)))

            # end63 = time()
            # print("6.3. [While]: Coeffs and constants constrains {}".format(int((end63 - start63) * 10**5) / 10**5))

            # end6 = time()
            # print("6. [While]: Compute gurobi init takes {}".format(int((end6 - start6) * 10**5) / 10**5))

            # start7 = time()

            # print(f"Optimizing model at time k = {k}")
            mod.optimize()
            # if not debugging:
            print(f"Gurobi status: {mod.Status}")

            # Print model statistics
            mod.printStats()
            # print(mod.Kappa)
                
            # end7 = time()
            # print("7. [While]: Compute gurobi optimization takes {}".format(int((end7 - start7) * 10**5) / 10**5))

            # start8 = time()

            if( mod.Status ==  gb.GRB.INFEASIBLE ):
                # model was infeasible
                mod.computeIIS()  # irreducible system of infeasible inequalities
                mod.write(f"LP_lineariz_IIS_k={k}.ilp")
                print("ERROR. Problem infeasible at time k={}. Halting...".format(k))
                mod.write(f"LP_lineariz_k={k}.lp")
                assert(False)
            
            
            if mod.Status == 5:
                print(mod.UnbdRay)
                for i in range(ut_dim * (T-k)):
                    if mod.UnbdRay[i] != 0:
                        print(f"Unbounded Ray! {mod.UnbdRay[i]}")
                        print(f"Index: {i}")
                        print(f"Unbounded ray has positive value for time: {i//ut_dim}")
                        print(f"Unbounded ray has positive value for group: {age_groups[(i%ut_dim) // num_controls]}")
                        print(f"Unbounded ray has positive value for control: {controls[(i%ut_dim) % num_controls]}")
            
            if mod.Kappa > 1000000000:
                mod.write(f"LP_lineariz_k={k}.lp")
                assert(False)

            # mod.write(f"LP_lineariz_k={k}.lp")
            # if not debugging:
            print(f"Objective value for Line heur k = {k}, it = {inner_iterations}: {mod.objVal}")

            negative_controls = (u_vars_vec.X<-1e-6)
            negative_controls_idx = np.where(negative_controls)
            for i in negative_controls_idx[0]:
                print(f"Negative controls! {u_vars_vec[i].X}")
                print(f"Index: {i}")
                print(f"Negative control is for time: {i//ut_dim}")
                print(f"Negative control is from group: {age_groups[(i%ut_dim) // num_controls]}")
                print(f"Negative control is for control: {controls[(i%ut_dim) % num_controls]}")

            assert (u_vars_vec.X >= -1e-6).all()

            if mod.Status != 2:
                print("Optimization did not finish correctly!")
                assert(False)

            step_shadow_prices = {}

            for con in range(num_constraints):
                step_shadow_prices[all_labels[con]] = all_const[con].Pi

            dynModel.shadowPrices[k] = step_shadow_prices

            # mod.write(f"LP_lineariz_k={k}.lp")

            # for i in range(len(uvars_opt)):
            #     for j in range(len(uvars_opt[i])):
            #         uvars_opt[i, j] = max(uvars_opt[i,j], 0)
            

            # extract decisions for current period (testing and alphas)
            uhat_seq = np.reshape(u_vars_vec.X, np.shape(obj_coefs), 'F')
            

            if optimize_bouncing is False:
                np_all_bouncing_idx_all_times = np.array(all_bouncing_idx_all_times)      
                uhat_seq[np_all_bouncing_idx_all_times%ut_dim, np_all_bouncing_idx_all_times//ut_dim] = 0
                
                Xhat_seq, uhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq, use_bounce_var)

            reward = get_real_reward(dynModel, uhat_seq)

            # if not debugging:
            print(f"Actual reward of current solution: {reward}")

            # Save best reward and decisions
            if reward > best_uhat_and_reward[1]:
                best_uhat_and_reward = (uhat_seq, reward)

            # Norm Infinity
            u_hat_lockdown_difference = min([max([abs(uhat_seq[i%ut_dim, i//ut_dim] - previous_u_hat_seq[0][i%ut_dim, i//ut_dim]) for i in all_lockdowns_idx_all_times]) + max([abs(uhat_seq[i%ut_dim, i//ut_dim] - previous_u_hat_seq[0][i%ut_dim, i//ut_dim]) / (dynModel.parameters['global-parameters']['C_atest'] if dynModel.parameters['global-parameters']['C_atest']>0 else 1e-5) for i in a_test_idx_all_times]) + max([abs(uhat_seq[i%ut_dim, i//ut_dim] - previous_u_hat_seq[0][i%ut_dim, i//ut_dim]) / (dynModel.parameters['global-parameters']['C_mtest'] if dynModel.parameters['global-parameters']['C_mtest']>0 else 1e-5) for i in m_test_idx_all_times]) for previous_u_hat_seq in all_u_hat_seq_and_rewards])

            

            # u_hat_lockwdown_difference = min(max([abs(uhat_seq[i, j] - uhat_seq[i, j]) for i in range(ut_dim) for j in range(T-k)]), abs(reward - all_u_hat_seq_and_rewards[-1][1]))

            # if not debugging:
            print(f"Minimum Difference with past solutions:{u_hat_lockdown_difference}.")

            if u_hat_lockdown_difference < 1e-6:
                # if not debugging:
                print(f"New solution equals old solution at k={k}")

            inner_iterations += 1

            all_u_hat_seq_and_rewards += [(uhat_seq, reward)]

            assert (uhat_seq>=-1e-6).all()
            
            # for ti in range(T-k):
            #     for act in activities:
            #         act_indices = slice(controls.index(act), ut_dim, num_controls)
            #         if (uvars_opt[act_indices, ti] >0).any():
            #             print(f"At time {ti+k} for subproblem {k} the activity {act} has non-zero lockdown.")

            # end8 = time()
            # print("8. [While]: Compute gurobi status takes {}".format(int((end8 - start8) * 10**5) / 10**5))
            # end_it = time()
            # print("9. [While]: Time taken by one iteration: {} seconds\n".format(end_it - start_it))

        # start9 = time()
        # uopt_seq[:,k] = best_uhat_and_reward[0][:,0]
        
        uk_opt_dict, alphak_opt_dict = buildAlphaDict(best_uhat_and_reward[0][:,0])

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
        if(use_bounce_var and optimize_bouncing):
            dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict, BH, BICU)
        else:
            dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict)

        # update uhat_sequence
        uhat_seq = best_uhat_and_reward[0][:, 1:]
        # end9 = time()
        # print("10. [For]: Postprocessing takes {}".format(int((end9 - start9) * 10**5) / 10**5))
            
        # print(f"u_optSeq at time {k} is {uopt_seq[:,k]}")
        # print(f"uhat_seq is {uhat_seq}")
        #
        # for ti in range(T-k):
        #     for act in activities:
        #         act_indices = slice(controls.index(act), ut_dim, num_controls)
        #         if (uvars_opt[act_indices, ti] >0).any():
        #             print(f"A time {ti+k} for subproblem {k} the activity {act} has non-zero lockdown.")

        # print(f"States at stage {k}")
        # print(dynModel.get_state(k))

    # print("uopt matrix is")
    # print(uopt_seq)

    return dynModel

####################################
# TESTING
# Global variables
# simulation_params = {
#         'dt':1.0,
#         'days': 182,
#         'region': "Ile-de-France",
#         'quar_freq': 182,
# }


# # Define time variables
# simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

# # Define mixing method
# mixing_method = {
#     "name":"mult",
#     "param_alpha":1.0,
#     "param_beta":0.5,
#     #"param":float(args.mixing_param) if args.mixing_param else 0.0,
# }

# # Read group parameters
# with open("../parameters/"+simulation_params["region"]+".yaml") as file:
#     # The FullLoader parameter handles the conversion from YAML
#     # scalar values to Python the dictionary format
#     universe_params = yaml.load(file, Loader=yaml.FullLoader)

# # Read initialization
# with open("../initialization/initialization.yaml") as file:
#     # The FullLoader parameter handles the conversion from YAML
#     # scalar values to Python the dictionary format
#     initialization = yaml.load(file, Loader=yaml.FullLoader)

# # Define policy
# with open('../benchmarks/static_infected_10.yaml') as file:
#     # The FullLoader parameter handles the conversion from YAML
#     # scalar values to Python the dictionary format
#     policy_file = yaml.load(file, Loader=yaml.FullLoader)
# alphas_vec = policy_file['alphas_vec']

# # Percentage infected at time 0
# perc_infected = 10
# # Move population to infected (without this there is no epidem.)
# for group in initialization:
# 	change = initialization[group]["S"]*perc_infected/100
# 	initialization[group]["S"] = initialization[group]["S"] - change
# 	initialization[group]["I"] = initialization[group]["I"] + change

# Create environment
# dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)

# Set up testing decisions: no testing for now
# a_tests_vec, m_tests_vec = no_tests(dynModel)
# tests = {
#     'a_tests_vec':a_tests_vec,
#     'm_tests_vec':m_tests_vec,
# }

# # Check reading of fundamental contact matrices
# ones_X = np.ones(9*13)
# ones_u = np.ones(9*10)
# J = get_Jacobian_u(dynModel,ones_X,ones_u,mixing_method)
# ag = 8
# print(J[ag*num_compartments + SEIR_groups.index('S_g'),:])

# #######################
# # Check the function that resets the simulation
# # Run the model for the whole time range
# for t in range(simulation_params['time_periods']):
#     dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], alphas_vec[t])

# # Print model stats
# print("Results from initial simulation:\n")
# dynModel.print_stats()

# # reset the model, Run it again and print stats
# dynModel.reset_time(5)
# for t in np.arange(5,simulation_params['time_periods'],1):
#     dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], alphas_vec[t])
# print("\nResults from second simulation:\n")
# dynModel.print_stats()

# #######################
# # Check Jacobians for compilation errors
# # X_hat = np.random.rand(num_age_groups*num_compartments)
# # u_hat = np.random.rand(num_age_groups*num_controls)
# #
# # get_Jacobian_X(dynModel,X_hat,u_hat,mixing_method)
# # get_Jacobian_u(dynModel,X_hat,u_hat,mixing_method)
