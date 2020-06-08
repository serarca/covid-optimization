# -*- coding: utf-8 -*-
import yaml
from inspect import getsourcefile
import os.path
import sys
import numpy as np
import pandas as pd
import math
import gurobipy as gb

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from group import *
from forecasting_heuristic import no_tests

age_groups = [ 'age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49', 'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus' ]
SEIR_groups = [ 'S_g', 'E_g', 'I_g', 'R_g', 'N_g', 'Ia_g', 'Ips_g', \
       'Ims_g', 'Iss_g', 'Rq_g', 'H_g', 'ICU_g', 'D_g' ]
activities = ['home','leisure','other','school','transport','work']
controls = [ 'Nmtest_g', 'Natest_g', 'BounceH_g', 'BounceICU_g' ]
controls.extend(activities)

num_age_groups = len(age_groups)
num_compartments = len(SEIR_groups)
num_controls = len(controls)
num_activities = len(activities)

#def get_index_X(ag_name, SEIRg_name, age_groups, SEIR_groups):
#    return age_groups.index(ag_name)*len(SEIR_groups) + SEIR_groups.index(SEIRg_name)

#def get_index_u(ag_name, SEIRg_name, age_groups, SEIR_groups):
#    return age_groups.index(ag_name)*len(SEIR_groups) + SEIR_groups.index(SEIRg_name)

####################################
# Calculate the Jacobian with respect to X (states)
def get_Jacobian_X(dynModel, X_hat, u_hat, mixing_method):
    """ Calculates the Jacobian for a given control trajectory u_hat and corresponding state trajectory X_hat """
    # This assumes that the order of components in X_hat and u_hat is as follows:
    # For X_hat: all the SEIR states for age group 1, then all the SEIR states for age group 2, etc.
    # For u_hat: all the controls for age group 1, then all the controls for age group 2, etc.
    # The order of the SEIR states is as follows:
    # X_hat = [ S_g, E_g, I_g, R_g, N_g, Ia_g, Ips_g, Ims_g, Iss_g, Rq_g, H_g, ICU_g, D_g , ...]
    # The order of the controls is as follows:
    # u_hat = [ Nmtest_g, Natest_g, BounceH_g, BounceICU_g, alpha_{g,activity1}, alpha_{g,activity2}, ... ]

    jacob = np.zeros((num_age_groups*num_compartments, num_age_groups*num_compartments))

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
        jacob[Sg_idx,Sg_idx] = - dynModel.groups[age_groups[ag]].parameters['beta'] * contacts

        # df^Sg/dNh all h
        for ah in range(0, num_age_groups):
            jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('N_g')] = dynModel.groups[age_groups[ag]].parameters['beta'] * X_hat[Sg_idx] * c_gh_arr[ah] * I_h_slice[ah] / ((N_h_slice[ah] + Rq_h_slice[ah])**2)

        # df^Sg/dIh all h
        for ah in range(0, num_age_groups):
            jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('I_g')] = - dynModel.groups[age_groups[ag]].parameters['beta'] * X_hat[Sg_idx] * c_gh_arr[ah] / (N_h_slice[ah] + Rq_h_slice[ah])

        # df^Sg/dRqh all h
        for ah in range(0, num_age_groups):
            jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')] = dynModel.groups[age_groups[ag]].parameters['beta'] * X_hat[Sg_idx] * c_gh_arr[ah] * I_h_slice[ah] / ((N_h_slice[ah] + Rq_h_slice[ah])**2)

        ########### f^Eg
        # df^Eg/dSg
        jacob[Eg_idx,Sg_idx] = - jacob[Sg_idx,Sg_idx]

        # df^Eg/dNh all h
        for ah in range(0, num_age_groups):
            jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('N_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('N_g')]

        # df^Eg/dIh all h
        for ah in range(0, num_age_groups):
            jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('I_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('I_g')]

        # df^Eg/dRqh all h
        for ah in range(0, num_age_groups):
            jacob[Eg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')] = - jacob[Sg_idx,ah * num_compartments + SEIR_groups.index('Rq_g')]

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


####################################
# Calculate the Jacobian with respect to u (decisions/controls)
def get_Jacobian_u(dynModel, X_hat, u_hat, mixing_method):
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

    jacob = np.zeros((num_age_groups*num_compartments,num_age_groups*num_controls))

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
            partial_contacts_g_array = np.zeros(num_age_groups)

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
                    partial_contacts_g_array[ah] = alpha*dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]]\
                        *(lga**(alpha-1))*(lha**(beta))
                    partial_contacts_h = beta*dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]]\
                        *(lga**(alpha))*(lha**(beta-1))
                if (mixing_method['name']=="maxmin"):
                    explga = math.exp(alpha * lga)
                    explha = math.exp(alpha * lha)
                    partial_contacts_g_array[ah] = dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]] \
                        * ((alpha * lga * explga * explha - alpha * explga * lha * explha + explga * explha + explga**2)/ ((explga + explha)**2))
                    partial_contacts_h = dynModel.groups[age_groups[ag]].contacts[activities[act-offset]][age_groups[ah]] \
                        * ((alpha * lha * explha * explga - alpha * explha * lga * explga + explha * explga + explha**2)/ ((explga + explha)**2))

                # S:
                jacob[Sg_idx,lha_idx] = -dynModel.groups[age_groups[ag]].parameters['beta']*X_hat[Sg_idx]*partial_contacts_h\
                    *X_hat[Ih_idx]/(X_hat[Nh_idx]+X_hat[Rqh_idx])
                # E:
                jacob[Eg_idx,lha_idx] = - jacob[Sg_idx,lha_idx]

            # S:
            jacob[Sg_idx,lga_idx] = -dynModel.groups[age_groups[ag]].parameters['beta']*X_hat[Sg_idx]*np.dot(rho_array,partial_contacts_g_array)
            # E:
            jacob[Eg_idx,lga_idx] = - jacob[Sg_idx,lga_idx]

    return jacob

####################################
# Build X_hat given a dynModel, a starting point k, and a
# sequence of controls u_hats
def get_X_hat_sequence(dynModel, k, u_hat_sequence):
    """Given a dynamical model, a starting point k, and the controls for time periods k to T-1 for tests and lockdowns, we re-start the dynamical model at time k, and then run it until time T with the controls in u_hat. 
    
    This produces the nominal trajectory X_hat_sequence. X_hat_sequence is a np.array of shape (num_compartments * num_age_groups, T-k+1), where each column represents the X_hat at time k, k+1,...
    
    This assumes that the dynamical model has already been run at least up to point k (it takes the states at time k as the starting points for the new nominal trajectory).

    We assume as well that u_hat_sequence is a 2-d numpy array with shape (num_controls * num_age_groups, T-k) with each column corresponding to a u_hat at time k, k+1,..., T-1. Hence, u_hat_sequence[:,k] gives the u_hat at time k.

    Note we are not using the bouncing variables in forecasting X_hat_sequence.

    At the moment we return only x_hat[k+1],... x_hat[T]

    """
    # Erase the states after k so as to reset the dyn model
    dynModel.reset_time(k)

    # The total time horizon for the dynamical model
    T = dynModel.time_steps

    X_hat_sequence = np.zeros((num_compartments * num_age_groups, T-k))

    for t in range(T-k):
        # Write the values of u_hat at time t in dict form

        u_hat_dict, alphas = buildAlphaDict(u_hat_sequence[:,t])
        #Create m and a tests in the format taken by dynModel
        m_tests = {}
        a_tests = {}
        for g in age_groups:
            m_tests[g] = u_hat_dict[g]['Nmtest_g']
            a_tests[g] = u_hat_dict[g]['Natest_g']

        #Using the controls u_hat for time t, take time step
        dynModel.take_time_step(m_tests, a_tests, alphas)

        state = dynModel.get_state(t + k + 1)

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

            X_hat_sequence[Sg_idx, t] = state[age_groups[ag]]['S']
            X_hat_sequence[Eg_idx, t] = state[age_groups[ag]]['E']
            X_hat_sequence[Ig_idx, t] = state[age_groups[ag]]['I']
            X_hat_sequence[Rg_idx, t] = state[age_groups[ag]]['R']
            X_hat_sequence[Ng_idx, t] = state[age_groups[ag]]['N']
            X_hat_sequence[Iag_idx, t] = state[age_groups[ag]]['Ia']
            X_hat_sequence[Ipsg_idx, t] = state[age_groups[ag]]['Ips']
            X_hat_sequence[Imsg_idx, t] = state[age_groups[ag]]['Ims']
            X_hat_sequence[Issg_idx, t] = state[age_groups[ag]]['Iss']
            X_hat_sequence[Rqg_idx, t] = state[age_groups[ag]]['Rq']
            X_hat_sequence[Hg_idx, t] = state[age_groups[ag]]['H']
            X_hat_sequence[ICUg_idx, t] = state[age_groups[ag]]['ICU']
            X_hat_sequence[Dg_idx, t] = state[age_groups[ag]]['D']

    # Erase the states after k so as to reset the dyn model
    dynModel.reset_time(k)

    return X_hat_sequence



####################################
# Build a dictionary for the state from a numpy array
def buildStateDict(X_hat_array):
    """Given a state vector X, builds a dictionary for all compartments that is compatible with DynModel"""

# function that builds a dictionary with decisions from a large numpy array (for a given period)
def buildAlphaDict(u_hat_array):
    """Given an array u_hat_array, builds a dictionary for all decisions that is compatible with DynModel"""
    u_hat_dict = {}
    alphas = {}
    for ag in range(0, num_age_groups):
        u_hat_dict[age_groups[ag]] = {}
        alphas[age_groups[ag]] = {}
        u_hat_dict[age_groups[ag]]['Nmtest_g'] = u_hat_array[ag * num_controls + controls.index('Nmtest_g')]
        u_hat_dict[age_groups[ag]]['Natest_g'] = u_hat_array[ag * num_controls + controls.index('Natest_g')]
        u_hat_dict[age_groups[ag]]['BounceH_g'] = u_hat_array[ag * num_controls + controls.index('BounceH_g')]
        u_hat_dict[age_groups[ag]]['BounceICU_g'] = u_hat_array[ag * num_controls + controls.index('BounceICU_g')]
        u_hat_dict[age_groups[ag]]['home'] = u_hat_array[ag * num_controls + controls.index('home')]
        u_hat_dict[age_groups[ag]]['leisure'] = u_hat_array[ag * num_controls + controls.index('leisure')]
        u_hat_dict[age_groups[ag]]['other'] = u_hat_array[ag * num_controls + controls.index('other')]
        u_hat_dict[age_groups[ag]]['school'] = u_hat_array[ag * num_controls + controls.index('school')]
        u_hat_dict[age_groups[ag]]['transport'] = u_hat_array[ag * num_controls + controls.index('transport')]
        u_hat_dict[age_groups[ag]]['work'] = u_hat_array[ag * num_controls + controls.index('work')]

        alphas[age_groups[ag]]['home'] = u_hat_array[ag * num_controls + controls.index('home')]
        alphas[age_groups[ag]]['leisure'] = u_hat_array[ag * num_controls + controls.index('leisure')]
        alphas[age_groups[ag]]['other'] = u_hat_array[ag * num_controls + controls.index('other')]
        alphas[age_groups[ag]]['school'] = u_hat_array[ag * num_controls + controls.index('school')]
        alphas[age_groups[ag]]['transport'] = u_hat_array[ag * num_controls + controls.index('transport')]
        alphas[age_groups[ag]]['work'] = u_hat_array[ag * num_controls + controls.index('work')]

    return u_hat_dict, alphas

####################################
# our internal function here to calculate the contacts of a given age group with all other age groups
def calcContacts(dynModel, alphas, mixing_method, ag):
    contacts_ag = np.zeros(num_age_groups)
    for h in range(0, num_age_groups):
        contacts_ag[h] = n_contacts(dynModel.groups[age_groups[ag]], dynModel.groups[age_groups[h]], alphas, mixing_method)

    return contacts_ag

####################################
# Calculate M and gamma
def calculate_M_gamma_and_eta(dynModel):
    """Calculates the matrix M and the vectors gamma and eta that yield the objective"""

    # M should have number of rows equal to the len(mu(t))
    # and number of columns equal to the len of X(t)
    M = np.zeros((num_age_groups*num_controls, num_age_groups*num_compartments))

    # Vector gamma should have len equal to the size of X(t)
    gamma = np.zeros(num_age_groups*num_compartments)

    # Vector eta should have len equal to the size of X(t)
    eta = np.zeros(num_age_groups*num_compartments)

    for ag in range(0,num_age_groups):
        # Get all the useful indices for the columns
        Sg_idx = ag*num_compartments + SEIR_groups.index('S_g')
        Eg_idx = ag*num_compartments + SEIR_groups.index('E_g')
        Rg_idx = ag*num_compartments + SEIR_groups.index('R_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')

        LWorkg_idx = ag*num_controls + controls.index('work')

        # Rename parameters to make expressions similar to the Latex
        theta = dynModel.groups[age_groups[ag]].economics['lockdown_fraction']
        v_NLg = dynModel.groups[age_groups[ag]].economics['work_value']
        v_Dg = dynModel.groups[age_groups[ag]].economics['death_value']

        # Matrix M should have only non-zero entries in the rows
        # corresponding to the lockdown decisions and the columns
        # corresponding to S_g E_g R_g and R^q_g
        # NOTICE THAT WE ASSUME HERE THAT R^q_g is in
        # no lockdown if we want to implement the M with R^q
        # having the same lockdown we should have
        # (1-theta) * v_NLg  as well in the
        # column corresponding to R^q_g
        M[LWorkg_idx, Sg_idx] = (1- theta) * v_NLg
        M[LWorkg_idx, Eg_idx] = (1- theta) * v_NLg
        M[LWorkg_idx, Rg_idx] = (1- theta) * v_NLg
        M[LWorkg_idx, Rqg_idx] = 0

        # Vector gamma should have only nonzero elements in the
        # columns corresponding to states S_g E_g R_g and R^q
        # NOTICE THAT WE ASSUME HERE THAT R^q_g is not in
        # lockdown if we want to implement the obj with R^q being
        # in lockdown, we should have v_NLg * theta in the
        # column corresponding to R^q_g
        gamma[Sg_idx] = v_NLg * theta
        gamma[Eg_idx] = v_NLg * theta
        gamma[Rg_idx] = v_NLg * theta
        gamma[Rqg_idx] = v_NLg

        # Vector eta should have only nonzero elements in the
        # columns corresponding to D_g. We assume here that D(0)=0
        # but this should not change the decisions of the heur.
        eta[Dg_idx] = -v_Dg

    return M, gamma, eta

####################################
# Calculate coefficients for the constraint
def calculate_H_constraint_coefs(dynModel):
    """Calculates the coefficient vectors a, b that yield the H constraint"""

    # The size of a is always the size of X(t)
    # The size of b is always the size of u(t)
    a = np.zeros(num_compartments * num_age_groups)
    b = np.zeros(num_controls * num_age_groups)

    for ag in range(0,num_age_groups):

        #Useful indices for the elements of a
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        # Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        # Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        # ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')

        # #Useful indices for the elements of b
        # BHg_idx = ag*num_controls + controls.index('BounceH_g')
        # BICUg_idx = ag*num_controls + controls.index('BounceICU_g')

        # # Useful coefficients for a and b
        # mu_g = dynModel.groups[age_groups[ag]].parameters['mu']
        # pICU_g = dynModel.groups[age_groups[ag]].parameters['p_ICU']
        # pH_g = dynModel.groups[age_groups[ag]].parameters['p_H']
        # lambda_H_R_g = dynModel.groups[age_groups[ag]].parameters['lambda_H_R']
        # lambda_H_D_g = dynModel.groups[age_groups[ag]].parameters['lambda_H_D']


        a[Hg_idx] = 1
        # a[Ig_idx] = mu_g * pH_g
        # a[Issg_idx] = mu_g * (pH_g / (pH_g + pICU_g))
        # a[Hg_idx] = (1 - lambda_H_R_g - lambda_H_D_g)

        # b[BHg_idx] = -1

    return a, b


def calculate_ICU_constraint_coefs(dynModel):
    """Calculates the coefficient vectors a, b that yield the ICU constraint"""

    # The size of a is always the size of X(t)
    # The size of b is always the size of u(t)
    a = np.zeros(num_compartments * num_age_groups)
    b = np.zeros(num_controls * num_age_groups)

    for ag in range(0,num_age_groups):
        #Useful indices for the elements of a
        # Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        # Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
        # Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')

        # #Useful indices for the elements of b
        # BHg_idx = ag*num_controls + controls.index('BounceH_g')
        # BICUg_idx = ag*num_controls + controls.index('BounceICU_g')

        # # Useful coefficients for a and b
        # mu_g = dynModel.groups[age_groups[ag]].parameters['mu']
        # pICU_g = dynModel.groups[age_groups[ag]].parameters['p_ICU']
        # pH_g = dynModel.groups[age_groups[ag]].parameters['p_H']
        # lambda_ICU_R_g = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_R']
        # lambda_ICU_D_g = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_D']


        a[ICUg_idx] = 1 
        # a[Ig_idx] = mu_g * pICU_g
        # a[Issg_idx] = mu_g * (pICU_g / (pH_g + pICU_g))
        # a[ICUg_idx] = (1 - lambda_ICU_R_g - lambda_ICU_D_g)

        # b[BICUg_idx] = -1

    return a, b

########################################
# Consider a linear dynamical system of the form X(k+1)=A(k) X(k) + B(k) u(k) + c(k)
# and a linear expression of the form a*X(t)+b*u(t) for some a,b row vectors of suitable dimension.
# This function returns the coefficients for all the decisions u(k),...,u(T)
# appearing in such an expression, for every period t=k,k+1,...,T
def calculate_all_constraint_coefs(dynModel, k, Xhat_seq, uhat_seq, a_matrix, b_matrix):
    """Get coefficients for decisions appearing in a generic linear constraint in each period k,k+1,...
    a_matrix and b_matrix are matrices, for now. Can change to dictionaries later. 
    a_matrix: rows = number of "types" of constraints, columns = num_compartments * num_age_groups
    b_matrix: rows = number of "types" of constraints, columns = num_controls * num_age_groups"""
    
    # shorthand for a few useful parameters
    T = dynModel.time_steps   
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = a_matrix.shape[0]

    assert( Xhat_seq.shape==(Xt_dim, T-k+1) )
    assert( uhat_seq.shape==(ut_dim, T-k+1) )
    assert(a_matrix.shape ==(num_constraints,Xt_dim))
    assert(b_matrix.shape ==(num_constraints,ut_dim))
    
    # Some pre-processing:
    # Calculate matrices A and B, and vector c, at given Xhat_seq and uhat_seq, across all the necessary time indices
    # Hold these as dictionaries, where the key is the time t.
    At = {}
    Bt = {}
    ct = {}
    for t in range(k,T+1):
        # get Xhat(t) and uhat(t)
        Xhat_t = Xhat_seq[:,t-k]
        uhat_t = uhat_seq[:,t-k]

        jacob_X = get_Jacobian_X(dynModel, Xhat_t, uhat_t, mixing_method)
        jacob_u = get_Jacobian_u(dynModel, Xhat_t, uhat_t, mixing_method)
        
        # Calculate linearization coefficients for X(t+1)
        At[t] = np.eye(Xt_dim) + dynModel.dt * jacob_X
        Bt[t] = dynModel.dt * jacob_u
        ct[t] = dynModel.dt * (get_F(dynModel, Xhat_t, uhat_t) - jacob_X @ Xhat_t - jacob_u @ uhat_t)

    # All constraint coefficients are stored in dictionary u_coeffs: u_coeffs has a key for each
    # period t in {k,k+1,...,T}. The value for key t stores, in turn, another dictionary, which holds the constraint coefficients
    # of the constraints indexed with t.
    # In that dictionary, the key is the index of a constraint "type", and the value is a 2D numpy array with
    # T-k+1 rows (one for every time period k, k+1, ..., T), and (ut_dim) columns. These are the coefficients for
    # all the controls u(k),...,u(T) appearing in the expression a*X(t) + b*u(t).
    u_coeffs = {}

    # The linear expression for a constraint also has constants, which we store in a separate dictionary: constants. 
    # The constants dictionary has a key for each period in {k,k+1,...,T}. The value for key t stores, in turn, another dictionary,
    # which holds the constants of the constraints indexed with t.
    # In that dictionary, the key is the index of a constraint "type", and the value is the constant corresponding to the specific 
    # constraint type index and time period.
    constants = {}
    
    # Initialize with zeros. (May want to try using sparse matrices here!)
    for t in np.arange(k,T+1):
        u_coeffs[t] = {}
        constants[t] = {}
        for constr_index in range(num_constraints):
            u_coeffs[t][constr_index] = np.zeros((T-k+1,ut_dim))

    # We keep track of certain partial products of matrices / vectors that are useful
    # NOTE. When comparing this with Overleaf, note that we are only keeping track of
    # the relevant matrices for the current period t (i.e, ignoring t-1,t-2,etc.)
    At_bar = {}
    Xt_bar = Xhat_seq[:,0]      # initialize with X(k)=Xhat(k)

    for t in range(k,T+1): # loop over times k, k+1, ..., T to model constraints indexed with t
        
        # Calculate constants for period t
        for constr_index in range(num_constraints):
            constants[t][constr_index] = a_matrix[constr_index,:] @ Xt_bar
        
        # Update auxiliary vector Xt_bar
        Xt_bar = At[t] @ Xt_bar + ct[t]

        # Calculate coefficients for all controls appearing in the constraint for period t
        # NOTE: The coefficients for control u(tau) are stored on row indexed (tau-k) of the 2D array
        for constr_index in range(num_constraints):
            # coefs for u[t]
            u_coeffs[t][constr_index][t-k,:] = b_matrix[constr_index,:]
        
        # Initialize At_bar for tau=t-1
        At_bar[t-1] = np.eye(Xt_dim,Xt_dim)   
            
        for tau in range(t-1,k-1,-1):
            for constr_index in range(num_constraints):
                # coefs for u[t-1], u[t-2], ..., u[k]
                u_coeffs[t][constr_index][tau-k,:] = a_matrix[constr_index,:] @ At_bar[tau] @ Bt[tau]
                
            # Update At_bar for next round
            At_bar[tau-1] = At_bar[tau] @ At[tau]

    return u_coeffs, constants


def calculate_BH_constraint_coefs(dynModel, group):
    """Calculates the coefficient vectors a, b that yield the BH constraint for group in age_groups. Observe that we have a different constraint for each group."""

    # The size of a is always the size of X(t)
    # The size of b is always the size of u(t)
    a = np.zeros(num_compartments * num_age_groups)
    b = np.zeros(num_controls * num_age_groups)

    ag = age_groups.index(group)

    #Useful indices for the elements of a
    Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
    Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
    Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
    ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')

    #Useful indices for the elements of b
    BHg_idx = ag*num_controls + controls.index('BounceH_g')
    BICUg_idx = ag*num_controls + controls.index('BounceICU_g')

    # Useful coefficients for a and b
    mu_g = dynModel.groups[age_groups[ag]].parameters['mu']
    pICU_g = dynModel.groups[age_groups[ag]].parameters['p_ICU']
    pH_g = dynModel.groups[age_groups[ag]].parameters['p_H']
    lambda_ICU_R_g = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_R']
    lambda_ICU_D_g = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_D']


    a[Ig_idx] = - mu_g * pH_g
    a[Issg_idx] = - mu_g * (pH_g / (pH_g + pICU_g))

    b[BHg_idx] = 1

    return a, b



def calculate_BICU_constraint_coefs(dynModel, group):
    """Calculates the coefficient vectors a, b that yield the BICU constraint for group in age_groups. Observe that we have a different constraint for each group."""

    # The size of a is always the size of X(t)
    # The size of b is always the size of u(t)
    a = np.zeros(num_compartments * num_age_groups)
    b = np.zeros(num_controls * num_age_groups)

    ag = age_groups.index(group)

    #Useful indices for the elements of a
    Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
    Ig_idx = ag*num_compartments + SEIR_groups.index('I_g')
    Issg_idx = ag*num_compartments + SEIR_groups.index('Iss_g')
    ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')

    #Useful indices for the elements of b
    BHg_idx = ag*num_controls + controls.index('BounceH_g')
    BICUg_idx = ag*num_controls + controls.index('BounceICU_g')

    # Useful coefficients for a and b
    mu_g = dynModel.groups[age_groups[ag]].parameters['mu']
    pICU_g = dynModel.groups[age_groups[ag]].parameters['p_ICU']
    pH_g = dynModel.groups[age_groups[ag]].parameters['p_H']
    lambda_ICU_R_g = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_R']
    lambda_ICU_D_g = dynModel.groups[age_groups[ag]].parameters['lambda_ICU_D']


    a[Ig_idx] = - mu_g * pICU_g
    a[Issg_idx] = - mu_g * (pICU_g / (pH_g + pICU_g))

    b[BICUg_idx] = 1

    return a, b


def calculate_Mtest_constraint_coefs(dynModel):
    """Calculates the coefficient vectors a, b that yield the Mtest constraint"""

    # The size of a is always the size of X(t)
    # The size of b is always the size of u(t)
    a = np.zeros(num_compartments * num_age_groups)
    b = np.zeros(num_controls * num_age_groups)

    for ag in range(0,num_age_groups):
        #Useful indices for the elements of b
        Nmtestg_idx = ag*num_controls + controls.index('Nmtest_g')

        b[Nmtestg_idx] = 1

    return a, b


def calculate_Atest_constraint_coefs(dynModel):
    """Calculates the coefficient vectors a, b that yield the Atest constraint"""

    # The size of a is always the size of X(t)
    # The size of b is always the size of u(t)
    a = np.zeros(num_compartments * num_age_groups)
    b = np.zeros(num_controls * num_age_groups)

    for ag in range(0,num_age_groups):
        #Useful indices for the elements of b
        Natestg_idx = ag*num_controls + controls.index('Natest_g')

        b[Natestg_idx] = 1

    return a, b



########### Returns f(X(t), u(t))  = X(t+1). Does not change
########### dynModel.
def get_F(dynModel, X, u):
    ''' Will return the next states given the state X and
    controls u. For this, it replaces the current state in
    dynModel for X, runs one step with controls u, extracts the
    resulting states, and re-starts the model to it's original
    state. Assumes as all functions above that X is ordered by
    compartment and then by group, and u by control and then by group.
    '''

    assert(X.shape == (num_compartments * num_age_groups, ))
    assert(u.shape == (num_controls * num_age_groups, ))

    # We save the initial time of dynModel to revert back to
    initial_time_of_model = dynModel.t
    initial_state_dict = dynModel.get_state(dynModel.t)

    #Determine the testing at time t given by u
    u_hat_dict, alphas = buildAlphaDict(u)

    m_tests = {}
    a_tests = {}
    for g in age_groups:
        m_tests[g] = u_hat_dict[g]['Nmtest_g']
        a_tests[g] = u_hat_dict[g]['Natest_g']

    # Add the current state to the dynModel
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
        Nmtestg_idx = ag*num_controls + controls.index('Nmtest_g')
        Natestg_idx = ag*num_controls + controls.index('Natest_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')

    # Run a step of the dyn model
    dynModel.take_time_step(m_tests, a_tests, alphas)

    # Get the current state (not sure if it shold be t or t-1)
    state_next_step = dynModel.get_state(dynModel.t)

    #X_next_step = np.zeros((1, num_compartments * num_age_groups))
    X_next_step = np.zeros(num_compartments * num_age_groups)

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
        Nmtestg_idx = ag*num_controls + controls.index('Nmtest_g')
        Natestg_idx = ag*num_controls + controls.index('Natest_g')
        Rqg_idx = ag*num_compartments + SEIR_groups.index('Rq_g')
        Hg_idx = ag*num_compartments + SEIR_groups.index('H_g')
        ICUg_idx = ag*num_compartments + SEIR_groups.index('ICU_g')
        Dg_idx = ag*num_compartments + SEIR_groups.index('D_g')

        X_next_step[Sg_idx] = state_next_step[age_groups[ag]]['S']
        X_next_step[Eg_idx] = state_next_step[age_groups[ag]]['E']
        X_next_step[Ig_idx] = state_next_step[age_groups[ag]]['I']
        X_next_step[Rg_idx] = state_next_step[age_groups[ag]]['R']
        X_next_step[Ng_idx] = state_next_step[age_groups[ag]]['N']
        X_next_step[Iag_idx] = state_next_step[age_groups[ag]]['Ia']
        X_next_step[Ipsg_idx] = state_next_step[age_groups[ag]]['Ips']
        X_next_step[Imsg_idx] = state_next_step[age_groups[ag]]['Ims']
        X_next_step[Issg_idx] = state_next_step[age_groups[ag]]['Iss']
        X_next_step[Rqg_idx] = state_next_step[age_groups[ag]]['Rq']
        X_next_step[Hg_idx] = state_next_step[age_groups[ag]]['H']
        X_next_step[ICUg_idx] = state_next_step[age_groups[ag]]['ICU']
        X_next_step[Dg_idx] = state_next_step[age_groups[ag]]['D']

    # Erase the states after t so as to reset the dyn model, also populate t with correct state
    dynModel.reset_time(initial_time_of_model)
    dynModel.write_state(dynModel.t, initial_state_dict)

    # Return dynModel to initial state
    return X_next_step



####################################
# This function creates the coefficients of u_t in the linearization of the objective
def create_lp_obj_coeffs(dynModel, mixing_method, t, M, gamma, eta, deltaT):
    T = dynModel.time_steps

    # A, B, c arrays/matrices. These are timeseries, e.g. A[t] is A_t. The length of the timeseries is
    A = np.zeros((num_age_groups * num_compartments, num_age_groups * num_compartments, T - t))
    B = np.zeros((num_age_groups * num_compartments, num_age_groups * num_controls, T - t))
    c = np.zeros((num_age_groups * num_compartments, T - t))

    # Time series of arrays containing the coefficients in front of u_t. u_coeff[tau - t] is coefficient for u_tau
    u_coeff = np.zeros((num_age_groups * num_controls, T - t - 1))

    # Calculate u_hat to be projected forward
    u_hat_forward_seq = np.zeros((num_age_groups * num_controls, T - t + 1))

    # Get timeseries of forecast states
    X_hat_time_series = get_X_hat_sequence(dynModel, t, u_hat_forward_seq)

    for tau in range(t, T):

        jacob_X = get_Jacobian_X(dynModel, X_hat_time_series[:, tau - t], u_hat_forward_seq[:, tau - t], mixing_method)
        jacob_u = get_Jacobian_u(dynModel, X_hat_time_series[:, tau - t], u_hat_forward_seq[:, tau - t], mixing_method)

        # Calculate linearization coefficients for X_t+1
        A[:, :, tau - t] = np.eye(num_age_groups * num_compartments) + deltaT * jacob_X
        B[:, :, tau - t] = deltaT * jacob_u
        FXhatuhat = get_F(dynModel, X_hat_time_series[:, tau - t], u_hat_forward_seq[:, tau - t])
        c[:, tau - t] = deltaT * (FXhatuhat - np.dot(jacob_X, X_hat_time_series[:, tau - t]) - np.dot(jacob_u, u_hat_forward_seq[:, tau - t]))

        # Calculate coefficient for decision u_t
        # Add each term
        u_coeff[:, tau - t] += np.dot(X_hat_time_series[:, tau - t].reshape(1, -1), np.transpose(M))[0, :]

        a_prod_mat = np.eye(num_age_groups * num_compartments, num_age_groups * num_compartments)
        for u in reversed(range(tau, T)):
            temp_arr = np.dot(u_hat_forward_seq[:, u].reshape(1, -1), M) + gamma
            if (u > tau + 1):
                a_prod_mat = np.dot(A[:, :, u], a_prod_mat)
            temp_arr = np.dot(np.dot(temp_arr, a_prod_mat), B[:, :,  tau - t])
        u_coeff[:, tau - t] += temp_arr[0, :]

        temp_arr = eta
        for u in (tau, T - 1):
            temp_arr = np.dot(temp_arr, A[:, :, u])
        temp_arr = np.dot(temp_arr, B[:, :, tau])
        u_coeff[:, tau - t] += temp_arr[0, :]

    return u_coeff

####################################
# Main function: runs the linearization heuristic
def run_heuristic_linearization(dynModel, mixing_method):
    """Run the heuristic based on linearization"""

    # shorthand for a few useful parameters
    T = dynModel.time_steps
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups

    # calculate M, gamma, eta
    M, gamma, eta = calculate_M_gamma_and_eta(dynModel)

    # hospital constraints
    aH, bH =  calculate_H_coefficients(dynModel)
    coefs_H = {}
    for ag in age_groups:
        coefs_H[ag] = calculate_BH_coefficients(dynModel,ag)

    # ICU constraints
    aICU, bICU = calculate_ICU_coefficients(dynModel)
    coefs_ICU = {}
    for ag in age_groups:
        coefs_ICU[ag] = calculate_ICU_coefficients(dynModel,ag)

    # pick a starting u_hat sequence; for now, no testing
    u_hat = np.zeros(T, ut_dim)

    for k in np.range(T):

        # calculate state trajectory X_hat
        X_hat = get_X_hat_sequence(dynModel, k, u_hat)
        assert( np.shape(X_hat) == (Xt_dim,T-k+1) )

        # get coefficients for all decisions, i.e., u_hat_sequence
        coefs_u_obj = create_lp_obj_coeffs(dynModel, mixing_method, k, X_hat, u_hat, M, gamma, eta)
        assert( np.shape(coefs_u_obj) == (ut_dim,T-k+1) )

        # create empty model
        mod = gb.Model("Linearization Heuristic")

        # add all decisions using matrix format, and also specify objective coefficients
        u_vars = mod.addMVar(np.shape(u_hat), obj=coefs_u_obj, name="u")

        # add hospital constraints
        #for t in np.range(0:T-k+1):
            #mod.addConstr( aH@_vars[:,t] + bH)


####################################
# TESTING
# Global variables
simulation_params = {
        'dt':1.0,
        'days': 182,
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
