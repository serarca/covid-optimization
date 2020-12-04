# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# from IPython import get_ipython

# %%
import pandas as pd
from datetime import datetime, timedelta
import numpy
from gurobipy import *


# %%
import numpy as np
import pandas as pd

import sys

# %%
lower_data = pd.read_excel("./parameter_fitting/ile-de-france_data_master.xlsx",sheet_name="SEIR_params_conf_range_lower")
upper_data = pd.read_excel("./parameter_fitting/ile-de-france_data_master.xlsx",sheet_name="SEIR_params_conf_range_upper")


# %%
# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29','age_group_30_39', 'age_group_40_49', 'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
econ_activities = ['transport', 'leisure', 'other']
cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips',            'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]


# %%
data = pd.read_csv("./parameter_fitting/donnees-hospitalieres-classe-age-covid19-2020-07-27-19h00.csv", sep=";")
data.head()


# %%
# Extract days 
days = data[data['reg']==11][data['cl_age90']==0].jour.values
days


# %%
beds_real = {
    age_groups[i]:data[data['reg']==11][data['cl_age90']==10*i+9].hosp.values for i in range(0,9)
}
beds_real['age_group_80_plus']+=data[data['reg']==11][data['cl_age90']==90].hosp.values
beds_real['total']=data[data['reg']==11][data['cl_age90']==0].hosp.values

icus_real = {
    age_groups[i]:data[data['reg']==11][data['cl_age90']==10*i+9].rea.values for i in range(0,9)
}
icus_real['age_group_80_plus']+=data[data['reg']==11][data['cl_age90']==90].rea.values
icus_real['total']=data[data['reg']==11][data['cl_age90']==0].rea.values

deaths_real = {
    age_groups[i]:data[data['reg']==11][data['cl_age90']==10*i+9].dc.values for i in range(0,9)
}
deaths_real['age_group_80_plus']+=data[data['reg']==11][data['cl_age90']==90].dc.values
deaths_real['total']=data[data['reg']==11][data['cl_age90']==0].dc.values


# %%
mult_deaths = np.sum(data[data['reg']==11][data['cl_age90']==0].hosp.values)/np.sum(data[data['reg']==11][data['cl_age90']==0].dc.values)
# print(mult_deaths)
mult_icus = np.sum(data[data['reg']==11][data['cl_age90']==0].hosp.values)/np.sum(data[data['reg']==11][data['cl_age90']==0].rea.values)
# print(mult_icus)

# %%
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
parentdir = os.path.dirname(current_dir)
sys.path.insert(0,parentdir) 
sys.path.insert(0, parentdir+"/heuristics")
sys.path.insert(0, parentdir+"/heuristics/LP-Models")
sys.path.insert(0, parentdir+"/fast_gradient")


from fast_group import FastDynamicalModel
from aux import *


# %%
# region = "Ile-de-France"
region = "fitted"

# %%
# Read group parameters
with open("./parameters/"+region+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)
    
# Read initialization
with open("./initialization/patient_zero.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    original_initialization = yaml.load(file, Loader=yaml.FullLoader)

# Read econ parameters
with open("./parameters/econ.yaml") as file:
    econ_params = yaml.load(file, Loader=yaml.FullLoader)

experiment_params = {
    'delta_schooling':0.5,
    'xi':0,
    'icus':3000,
}


# %%
date_1 = datetime.strptime("2020-03-17", '%Y-%m-%d')
date_2 = datetime.strptime("2020-04-14", '%Y-%m-%d')
date_3 = datetime.strptime("2020-05-11", '%Y-%m-%d')
date_4 = datetime.strptime("2020-06-02", '%Y-%m-%d')
date_5 = datetime.strptime("2020-06-15", '%Y-%m-%d')
date_6 = datetime.strptime("2020-06-22", '%Y-%m-%d')
date_7 = datetime.strptime("2020-07-06", '%Y-%m-%d')
#final_date = datetime.strptime(days[-1], '%Y-%m-%d') + timedelta(days=1)
final_date = datetime.strptime("2020-07-27", '%Y-%m-%d')


# %%
print("Days {}".format((final_date-date_1).days+30))


# %%
from copy import deepcopy


# %%
# Create model
mixing_method = {}
eta_init = 0.0353893527721194
dynModel = FastDynamicalModel(universe_params, econ_params, experiment_params, 1, mixing_method, (final_date-date_1).days, 0, eta_init)


# %%
params = pd.read_excel("./parameter_fitting/ile-de-france_data_master.xlsx",sheet_name="SEIR_params", index_col = 0)
initial_params = {
    "mu":params['mu'].values,
    "sigma":params['sigma'].values,
    "p_ICU":params['p_ICU'].values,
    "p_H":params['p_H'].values,
    "lambda_H_R":params['lambda_HR'].values,
    "lambda_H_D":params['lambda_HD'].values,
    "lambda_ICU_R":params['lambda_ICUR'].values,
    "lambda_ICU_D":params['lambda_ICUD'].values
}

params = pd.read_excel("./parameter_fitting/ile-de-france_data_master.xlsx",sheet_name="SEIR_params_conf_range_lower", index_col = 0)
lower_params = {
    "mu":params['mu'].values,
    "sigma":params['sigma'].values,
    "p_ICU":params['p_ICU'].values,
    "p_H":params['p_H'].values,
    "lambda_H_R":params['lambda_HR'].values,
    "lambda_H_D":params['lambda_HD'].values,
    "lambda_ICU_R":params['lambda_ICUR'].values,
    "lambda_ICU_D":params['lambda_ICUD'].values
}

params = pd.read_excel("./parameter_fitting/ile-de-france_data_master.xlsx",sheet_name="SEIR_params_conf_range_upper", index_col = 0)
upper_params = {
    "mu":params['mu'].values,
    "sigma":params['sigma'].values,
    "p_ICU":params['p_ICU'].values,
    "p_H":params['p_H'].values,
    "lambda_H_R":params['lambda_HR'].values,
    "lambda_H_D":params['lambda_HD'].values,
    "lambda_ICU_R":params['lambda_ICUR'].values,
    "lambda_ICU_D":params['lambda_ICUD'].values
}


# %%
# Construct the windows for the parameters to move
windows = {}
for p in initial_params:
    windows[p] = (
        np.min(upper_params[p]/initial_params[p]),
        np.max(lower_params[p]/initial_params[p]),
    )
    
windows['beta_mixing'] = (0.1,4.0)
windows['alpha_mixing'] = (0.1,4.0)
windows['gamma_mixing'] = (0.5,2.0)


# %%
import copy
import math
best_v = 0
best_error = float('inf')
def error(v):

    days_ahead = v[0]
    alpha_mixing = v[1]
    beta_mixing = v[2]
    gamma_mixing_before = v[3]
    gamma_mixing_after = v[3]*(1-v[4])
    gamma_change = v[5] + v[0]


    upper_bound_home = 1.0
    upper_bound_leisure = 1.0
    upper_bound_other = 1.0
    upper_bound_school = 1.0
    upper_bound_work = 1.0
    upper_bound_transport = 1.0
    
    leisure_v = v[12:29]
    leisure_1 = leisure_v[0]
    leisure_2 = leisure_v[1]
    leisure_3 = leisure_v[2]
    leisure_4 = leisure_v[3]
    leisure_5 = leisure_v[4]
    leisure_6 = leisure_v[5]
    leisure_7 = leisure_v[6]
    leisure_denom = leisure_1+leisure_2+leisure_3+leisure_4+leisure_5+leisure_6+leisure_7

    other_v = v[19:26]
    other_1 = other_v[0]
    other_2 = other_v[1]
    other_3 = other_v[2]
    other_4 = other_v[3]
    other_5 = other_v[4]
    other_6 = other_v[5]
    other_7 = other_v[6]
    other_denom = other_1+other_2+other_3+other_4+other_5+other_6+other_7

    work_v = v[26:33]
    work_1 = work_v[0]
    work_2 = work_v[1]
    work_3 = work_v[2]
    work_4 = work_v[3]
    work_5 = work_v[4]
    work_6 = work_v[5]
    work_7 = work_v[6]
    work_denom = work_1+work_2+work_3+work_4+work_5+work_6+work_7
    
    transport_v = v[33:40]
    transport_1 = transport_v[0]
    transport_2 = transport_v[1]
    transport_3 = transport_v[2]
    transport_4 = transport_v[3]
    transport_5 = transport_v[4]
    transport_6 = transport_v[5]
    transport_7 = transport_v[6]
    transport_denom = transport_1+transport_2+transport_3+transport_4+transport_5+transport_6+transport_7


    # Construct initialization
    initialization = copy.deepcopy(original_initialization)
    for i,group in enumerate(age_groups):
        if group == "age_group_40_49":
            initialization[group]["I"] = initialization[group]["I"] + 1
            initialization[group]["S"] = initialization[group]["S"] - 1
        initialization[group]["N"] = initialization[group]["S"] + initialization[group]["E"] + initialization[group]["I"] + initialization[group]["R"]


    # Alphas
    a_before_date_1 = {
        'home':upper_bound_home,
        'leisure':upper_bound_leisure,
        'other':upper_bound_other,
        'school':upper_bound_school,
        'transport':upper_bound_transport,
        'work':upper_bound_work
    }
    a_between_date_1_2 = {
        'home':upper_bound_home,
        'leisure':leisure_1/leisure_denom*upper_bound_leisure,
        'other':other_1/other_denom*upper_bound_other,
        'school':0,
        'transport':transport_1/transport_denom*upper_bound_transport,
        'work':work_1/work_denom*upper_bound_work
    }
    a_between_date_2_3 = {
        'home':upper_bound_home,
        'leisure':leisure_1/leisure_denom*upper_bound_leisure,
        'other':other_1/other_denom*upper_bound_other,
        'school':0,
        'transport':transport_1/transport_denom*upper_bound_transport,
        'work':work_1/work_denom*upper_bound_work
    }
    a_between_date_3_4 = {
        'home':upper_bound_home,
        'leisure':(leisure_1+leisure_2)/leisure_denom*upper_bound_leisure,
        'other':(other_1+other_2)/other_denom*upper_bound_other,
        'school':0.33*upper_bound_school,
        'transport':(transport_1+transport_2)/transport_denom*upper_bound_transport,
        'work':(work_1+work_2)/work_denom*upper_bound_work
    }
    a_between_date_4_5 = {
        'home':upper_bound_home,
        'leisure':(leisure_1+leisure_2+leisure_3)/leisure_denom*upper_bound_leisure,
        'other':(other_1+other_2+other_3)/other_denom*upper_bound_other,
        'school':0.67*upper_bound_school,
        'transport':(transport_1+transport_2+transport_3)/transport_denom*upper_bound_transport,
        'work':(work_1+work_2+work_3)/work_denom*upper_bound_work
    }
    a_between_date_5_6 = {
        'home':upper_bound_home,
        'leisure':(leisure_1+leisure_2+leisure_3+leisure_4)/leisure_denom*upper_bound_leisure,
        'other':(other_1+other_2+other_3+other_4)/other_denom*upper_bound_other,
        'school':0.67*upper_bound_school,
        'transport':(transport_1+transport_2+transport_3+transport_4)/transport_denom*upper_bound_transport,
        'work':(work_1+work_2+work_3+work_4)/work_denom*upper_bound_work
    }
    a_between_date_6_7 = {
        'home':upper_bound_home,
        'leisure':(leisure_1+leisure_2+leisure_3+leisure_4+leisure_5)/leisure_denom*upper_bound_leisure,
        'other':(other_1+other_2+other_3+other_4+other_5)/other_denom*upper_bound_other,
        'school':1.0*upper_bound_school,
        'transport':(transport_1+transport_2+transport_3+transport_4+transport_5)/transport_denom*upper_bound_transport,
        'work':(work_1+work_2+work_3+work_4+work_5)/work_denom*upper_bound_work
    }
    a_after_date_7 = {
        'home':upper_bound_home,
        'leisure':(leisure_1+leisure_2+leisure_3+leisure_4+leisure_5+leisure_6)/leisure_denom*upper_bound_leisure,
        'other':(other_1+other_2+other_3+other_4+other_5+other_6)/other_denom*upper_bound_other,
        'school':1.0*upper_bound_school,
        'transport':(transport_1+transport_2+transport_3+transport_4+transport_5+transport_6)/transport_denom*upper_bound_transport,
        'work':(work_1+work_2+work_3+work_4+work_5+work_6)/work_denom*upper_bound_work
    }

    # Determine mixing method
    mixing_method_before = {
        "name":"mult",
        "param_alpha":alpha_mixing,
        "param_beta":beta_mixing,
        "fixed_gamma":gamma_mixing_before,
    }

    # Determine mixing method
    mixing_method_after = {
        "name":"mult",
        "param_alpha":alpha_mixing,
        "param_beta":beta_mixing,
        "fixed_gamma":gamma_mixing_after,
    }
    #dynModel.mixing_method = mixing_method_after

    # Number of days
    days_before_date_1 = int(days_ahead)
    days_between_dates_1_2 = (date_2-date_1).days
    days_between_dates_2_3 = (date_3-date_2).days
    days_between_dates_3_4 = (date_4-date_3).days
    days_between_dates_4_5 = (date_5-date_4).days
    days_between_dates_5_6 = (date_6-date_5).days
    days_between_dates_6_7 = (date_7-date_6).days

    days_after_date_7 = (final_date-date_7).days
    total_days = days_before_date_1 + days_between_dates_1_2 + days_between_dates_2_3 + days_between_dates_3_4 + days_between_dates_4_5+ days_between_dates_5_6+ days_between_dates_6_7 + days_after_date_7



    # Calculate alphas
    alphas_vec = []
    for t in range(days_before_date_1):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_before_date_1
        alphas_vec.append(alphas)
    for t in range(days_between_dates_1_2):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_between_date_1_2
        alphas_vec.append(alphas)
    for t in range(days_between_dates_2_3):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_between_date_2_3
        alphas_vec.append(alphas)
    for t in range(days_between_dates_3_4):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_between_date_3_4
        alphas_vec.append(alphas)
    for t in range(days_between_dates_4_5):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_between_date_4_5
        alphas_vec.append(alphas)
    for t in range(days_between_dates_5_6):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_between_date_5_6
        alphas_vec.append(alphas)
    for t in range(days_between_dates_6_7):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_between_date_6_7
        alphas_vec.append(alphas)
    for t in range(days_after_date_7):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_after_date_7
        alphas_vec.append(alphas)

    mixing_vec = []
    for t in range(int(gamma_change)):
        mixing_vec.append(mixing_method_before)
    for t in range(int(gamma_change),total_days):
        mixing_vec.append(mixing_method_after)
        
        
        
        
    # Calculate tests
    tests = np.zeros(len(age_groups))


    # Run model
    model_data_beds = {ag:[] for ag in age_groups+["total"]}
    model_data_icus = {ag:[] for ag in age_groups+["total"]}
    model_data_deaths = {ag:[] for ag in age_groups+["total"]}

    state = state_to_matrix(initialization)
    t_beds = 0
    t_icus = 0
    t_deaths = 0
    for i,ag in enumerate(age_groups):
        state_H = state[i,cont.index("H")]
        state_ICU = state[i,cont.index("ICU")]
        state_D = state[i,cont.index("D")]
        model_data_beds[ag].append(state_H)
        model_data_icus[ag].append(state_ICU)
        model_data_deaths[ag].append(state_D)
        t_beds+= state_H
        t_icus+= state_ICU
        t_deaths+= state_D
    model_data_beds["total"].append(t_beds)
    model_data_icus["total"].append(t_icus)
    model_data_deaths["total"].append(t_deaths)

    recalc_days = [0,
                 days_before_date_1,
                 days_before_date_1+days_between_dates_1_2,
                 days_before_date_1+days_between_dates_1_2+days_between_dates_2_3,
                 days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4,
                 days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5,
                 days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5+days_between_dates_5_6,
                 days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5+days_between_dates_5_6+days_between_dates_6_7,
                int(gamma_change)
                ]

    for t in range(total_days):
        if t in recalc_days:
            update_contacts = True
        else:
            update_contacts = False
        dynModel.mixing_method = mixing_vec[t]
        state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t, update_contacts=update_contacts)
        t_beds = 0
        t_icus = 0
        t_deaths = 0
        for i,ag in enumerate(age_groups):
            state_H = state[i,cont.index("H")]
            state_ICU = state[i,cont.index("ICU")]
            state_D = state[i,cont.index("D")]
            model_data_beds[ag].append(state_H)
            model_data_icus[ag].append(state_ICU)
            model_data_deaths[ag].append(state_D)
            t_beds+= state_H
            t_icus+= state_ICU
            t_deaths+= state_D
        model_data_beds["total"].append(t_beds)
        model_data_icus["total"].append(t_icus)
        model_data_deaths["total"].append(t_deaths)


    initial_date = date_1-timedelta(days=days_before_date_1)

    # Calculate the days of the model
    days_model = [initial_date+timedelta(days = t) for t in range(total_days + 1)]

    # Indices where to put the real data
    indices = [(datetime.strptime(d, '%Y-%m-%d') - initial_date).days for d in days]

    # Real data
    real_data_beds = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}
    real_data_icus = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}
    real_data_deaths = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}

    for k,ind in enumerate(indices):
        for ag in age_groups+["total"]:
            real_data_beds[ag][ind] = beds_real[ag][k]
            real_data_icus[ag][ind] = icus_real[ag][k]
            real_data_deaths[ag][ind] = deaths_real[ag][k]


    error_beds = 0
    error_icus = 0
    error_deaths = 0
    for ag in age_groups:
        error_beds += np.nanmean(np.abs(np.array(model_data_beds[ag])-np.array(real_data_beds[ag])))
        error_icus += np.nanmean(np.abs(np.array(model_data_icus[ag])-np.array(real_data_icus[ag])))
        error_deaths += np.nanmean(np.abs(np.array(model_data_deaths[ag])-np.array(real_data_deaths[ag])))
    error_beds_total = np.nanmean(np.abs(np.array(model_data_beds["total"])-np.array(real_data_beds["total"])))
    error_icus_total = np.nanmean(np.abs(np.array(model_data_icus["total"])-np.array(real_data_icus["total"])))
    error_deaths_total = np.nanmean(np.abs(np.array(model_data_deaths["total"])-np.array(real_data_deaths["total"])))

    diff = np.array(model_data_beds["total"])-np.array(real_data_beds["total"])
    error_beds_above = np.nanmean([max(d,0) for d in diff])
    error_beds_below = -np.nanmean([min(d,0) for d in diff])

    cumm_beds_model = [sum([model_data_beds["total"][k] for k in range(i+1) if not math.isnan(real_data_beds["total"][k])]) for i in range(len(model_data_beds["total"]))]
    cumm_beds_real = [sum([real_data_beds["total"][k] for k in range(i+1) if not math.isnan(real_data_beds["total"][k])]) for i in range(len(real_data_beds["total"]))]
    diff_cumm = np.array(cumm_beds_model)-np.array(cumm_beds_real)
    error_cumm_above = np.nanmean([max(d,0) for d in diff_cumm])
    error_cumm_below = -np.nanmean([min(d,0) for d in diff_cumm])



    #     error = error_beds_total
    #     error = mult_icus*error_icus_total
    #     error = mult_deaths*error_deaths_total
    upper_model_data = model_data_beds["total"]
    upper_days_model = days_model
    upper_real_data = real_data_beds["total"]
    error = error_beds+5*error_beds_total



    
    
    global best_error
    global best_v
    if error<best_error:
        best_error = error
        # print(best_error)
        # print("error_groups",error_beds)
        # print("error_total",error_beds_total)
        best_v = v
        # print(v)
        # plt.figure(1)
        # plt.plot(upper_days_model, upper_model_data, label="Upper")
        # plt.plot(upper_days_model, upper_real_data, label="Model L1")
        # plt.legend(loc='upper right')
        # plt.show()
        # print(a_before_date_1)
        # print(a_between_date_1_2)
        # print(a_between_date_2_3)
        # print(a_between_date_3_4)
        # print(a_between_date_4_5)
        # print(a_between_date_5_6)
        # print(a_between_date_6_7)
        # print(a_after_date_7)
        

        # Calculate l-april and l-may
        l_april = a_between_date_1_2
        l_may = {}
        for a in a_between_date_1_2.keys():
            l_may[a] = a_between_date_1_2[a]/3.0 + a_between_date_3_4[a]*2.0/3.0

        eq_activities = ['leisure','other','school','transport']
        t = 0.5
        nu = (1-t)*(1-l_may["work"])/(1-0.5851)
        eta = t*(1-np.mean([l_may[act] for act in eq_activities]))/(1-0.5851)
        gamma = 1-nu-eta
        # print(nu,eta,gamma)

    return error




# %%
# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
from scipy.optimize import minimize, Bounds, shgo, differential_evolution, dual_annealing
epsilon = 0.1
print("Best V 1 {}".format(best_v))

result = differential_evolution(error, [(50,140),
                                        windows['alpha_mixing'],windows['beta_mixing'],windows['gamma_mixing'],(0,0.5),(-50,50)]+
                                        [(0.8,1),(0.8,1),(0.8,1),(0.8,1),(0.8,1),(0.8,1)]+
                                        [(0,1.0),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)]*4)
                            # ,maxiter=0, popsize = 0, tol=1000000000, disp=True)
print("Best V 2 {}".format(best_v))


# %%
# v0 = (list(best_v[0:4]) + list(best_v[4]*initial_params["mu"]) + list(best_v[5]*initial_params['sigma']) + list(best_v[6]*initial_params['p_H'])+
#       list(best_v[7]*initial_params['p_ICU'])+list(best_v[8]*initial_params['lambda_H_R'])+list(best_v[9]*initial_params['lambda_H_D'])+
#     list(best_v[10]*initial_params['lambda_ICU_R'])+list(best_v[11]*initial_params['lambda_ICU_D']) + list(best_v[12:]))
v0 = best_v
error(best_v)


# %%
# best_v = [55.76574406,  1.65994558,  3.54471426,  1.81803306,  0.29064393, 10.16456894,
#   0.99267626,  0.89923718,  0.88046196,  0.97375347,  0.96930897,  0.90837394,
#   0.97350105,  0.55814137,  0.2174009 ,  0.36999562,  0.57774866,  0.95434156,
#   0.53047664,  0.85170554,  0.72113671,  0.49993411,  0.29182439,  0.84573728,
#   0.19667646,  0.34152441,  0.26903685,  0.42398907,  0.50355682,  0.30630559,
#   0.8104787 ,  0.58992106,  0.86364   ,  0.47907731,  0.63184319,  0.52658515,
#   0.59142121,  0.49368035,  0.44186236,  0.24121374]


# %%
lb= ([50,windows['alpha_mixing'][0],windows['beta_mixing'][0],windows['gamma_mixing'][0],0,-50]+
                [0.8,0.8,0.8,0.8,0.8,0.8]+[0,0,0,0,0,0,0]*4)
ub = ([140,windows['alpha_mixing'][1],windows['beta_mixing'][1],windows['gamma_mixing'][1],0.5,50]+
                [1,1,1,1,1,1]+[1,1,1,1,1,1,1]*4)
for i in range(len(v0)):
    assert(v0[i]>=lb[i])
    assert(ub[i]>=v0[i])


# %%
from scipy.optimize import minimize, Bounds, shgo, differential_evolution

bounds = Bounds(lb,ub)

print("Best V 3 {}".format(best_v))

result = minimize(error, v0, bounds = bounds)

print("Best V 4 {}".format(best_v))

# %%
# best_v


# %%
# best_v = [55.76574406,  1.67438361,  3.54751353,  1.81829588,  0.29775115,
#        10.16456894,  0.99615458,  0.89687269,  0.8787483 ,  0.96537944,
#         0.96665457,  0.90696783,  0.97371304,  0.56095857,  0.22207873,
#         0.36743667,  0.57386786,  0.94824818,  0.52793675,  0.8530484 ,
#         0.72332039,  0.50297928,  0.28944408,  0.83974079,  0.19482276,
#         0.34017182,  0.27339658,  0.42739236,  0.50644834,  0.31035431,
#         0.80520744,  0.58609923,  0.86516311,  0.48232211,  0.6341458 ,
#         0.52950161,  0.58770836,  0.49053886,  0.43910603,  0.2399934 ]


# %%
import copy
import math
# def plot_model(v, days_diff, gamma_mult, alphas_mult):

#     days_ahead = v[0]+days_diff
#     alpha_mixing = v[1]
#     beta_mixing = v[2]
#     gamma_mixing_before = v[3]*gamma_mult
#     gamma_mixing_after = v[3]*(1-v[4])*gamma_mult
#     gamma_change = v[5] + v[0]


#     upper_bound_home = v[6]
#     upper_bound_leisure = v[7]
#     upper_bound_other = v[8]
#     upper_bound_school = v[9]
#     upper_bound_work = v[10]
#     upper_bound_transport = v[11]
    
#     leisure_v = v[12:29]
#     leisure_1 = leisure_v[0]
#     leisure_2 = leisure_v[1]
#     leisure_3 = leisure_v[2]
#     leisure_4 = leisure_v[3]
#     leisure_5 = leisure_v[4]
#     leisure_6 = leisure_v[5]
#     leisure_7 = leisure_v[6]
#     leisure_denom = leisure_1+leisure_2+leisure_3+leisure_4+leisure_5+leisure_6+leisure_7

#     other_v = v[19:26]
#     other_1 = other_v[0]
#     other_2 = other_v[1]
#     other_3 = other_v[2]
#     other_4 = other_v[3]
#     other_5 = other_v[4]
#     other_6 = other_v[5]
#     other_7 = other_v[6]
#     other_denom = other_1+other_2+other_3+other_4+other_5+other_6+other_7

#     work_v = v[26:33]
#     work_1 = work_v[0]
#     work_2 = work_v[1]
#     work_3 = work_v[2]
#     work_4 = work_v[3]
#     work_5 = work_v[4]
#     work_6 = work_v[5]
#     work_7 = work_v[6]
#     work_denom = work_1+work_2+work_3+work_4+work_5+work_6+work_7
    
#     transport_v = v[33:40]
#     transport_1 = transport_v[0]
#     transport_2 = transport_v[1]
#     transport_3 = transport_v[2]
#     transport_4 = transport_v[3]
#     transport_5 = transport_v[4]
#     transport_6 = transport_v[5]
#     transport_7 = transport_v[6]
#     transport_denom = transport_1+transport_2+transport_3+transport_4+transport_5+transport_6+transport_7


#     # Construct initialization
#     initialization = copy.deepcopy(original_initialization)
#     for i,group in enumerate(age_groups):
#         if group == "age_group_40_49":
#             initialization[group]["I"] = initialization[group]["I"] + 1
#             initialization[group]["S"] = initialization[group]["S"] - 1
#         initialization[group]["N"] = initialization[group]["S"] + initialization[group]["E"] + initialization[group]["I"] + initialization[group]["R"]


#     # Alphas
#     a_before_date_1 = {
#         'home':upper_bound_home*alphas_mult,
#         'leisure':upper_bound_leisure*alphas_mult,
#         'other':upper_bound_other*alphas_mult,
#         'school':upper_bound_school*alphas_mult,
#         'transport':delta_transport*upper_bound_work*alphas_mult,
#         'work':upper_bound_work*alphas_mult
#     }
#     a_between_date_1_2 = {
#         'home':upper_bound_home*alphas_mult,
#         'leisure':leisure_1/leisure_denom*upper_bound_leisure*alphas_mult,
#         'other':other_1/other_denom*upper_bound_other*alphas_mult,
#         'school':0,
#         'transport':delta_transport*work_1/work_denom*upper_bound_work*alphas_mult,
#         'work':work_1/work_denom*upper_bound_work*alphas_mult
#     }
#     a_between_date_2_3 = {
#         'home':upper_bound_home*alphas_mult,
#         'leisure':leisure_1/leisure_denom*upper_bound_leisure*alphas_mult,
#         'other':other_1/other_denom*upper_bound_other*alphas_mult,
#         'school':0,
#         'transport':delta_transport*work_1/work_denom*upper_bound_work*alphas_mult,
#         'work':work_1/work_denom*upper_bound_work*alphas_mult
#     }
#     a_between_date_3_4 = {
#         'home':upper_bound_home*alphas_mult,
#         'leisure':(leisure_1+leisure_2)/leisure_denom*upper_bound_leisure*alphas_mult,
#         'other':(other_1+other_2)/other_denom*upper_bound_other*alphas_mult,
#         'school':0.33*delta_school*upper_bound_school*alphas_mult,
#         'transport':delta_transport*(work_1+work_2)/work_denom*upper_bound_work*alphas_mult,
#         'work':(work_1+work_2)/work_denom*upper_bound_work*alphas_mult
#     }
#     a_between_date_4_5 = {
#         'home':upper_bound_home*alphas_mult,
#         'leisure':(leisure_1+leisure_2+leisure_3)/leisure_denom*upper_bound_leisure*alphas_mult,
#         'other':(other_1+other_2+other_3)/other_denom*upper_bound_other*alphas_mult,
#         'school':0.67*delta_school*upper_bound_school*alphas_mult,
#         'transport':delta_transport*(work_1+work_2+work_3)/work_denom*upper_bound_work*alphas_mult,
#         'work':(work_1+work_2+work_3)/work_denom*upper_bound_work*alphas_mult
#     }
#     a_between_date_5_6 = {
#         'home':upper_bound_home*alphas_mult,
#         'leisure':(leisure_1+leisure_2+leisure_3+leisure_4)/leisure_denom*upper_bound_leisure*alphas_mult,
#         'other':(other_1+other_2+other_3+other_4)/other_denom*upper_bound_other*alphas_mult,
#         'school':0.67*delta_school*upper_bound_school*alphas_mult,
#         'transport':delta_transport*(work_1+work_2+work_3+work_4)/work_denom*upper_bound_work*alphas_mult,
#         'work':(work_1+work_2+work_3+work_4)/work_denom*upper_bound_work*alphas_mult
#     }
#     a_between_date_6_7 = {
#         'home':upper_bound_home*alphas_mult,
#         'leisure':(leisure_1+leisure_2+leisure_3+leisure_4+leisure_5)/leisure_denom*upper_bound_leisure*alphas_mult,
#         'other':(other_1+other_2+other_3+other_4+other_5)/other_denom*upper_bound_other*alphas_mult,
#         'school':1.0*delta_school*upper_bound_school*alphas_mult,
#         'transport':delta_transport*(work_1+work_2+work_3+work_4+work_5)/work_denom*upper_bound_work*alphas_mult,
#         'work':(work_1+work_2+work_3+work_4+work_5)/work_denom*upper_bound_work*alphas_mult
#     }
#     a_after_date_7 = {
#         'home':upper_bound_home*alphas_mult,
#         'leisure':(leisure_1+leisure_2+leisure_3+leisure_4+leisure_5+leisure_6)/leisure_denom*upper_bound_leisure*alphas_mult,
#         'other':(other_1+other_2+other_3+other_4+other_5+other_6)/other_denom*upper_bound_other*alphas_mult,
#         'school':1.0*delta_school*upper_bound_school*alphas_mult,
#         'transport':delta_transport*(work_1+work_2+work_3+work_4+work_5+work_6)/work_denom*upper_bound_work*alphas_mult,
#         'work':(work_1+work_2+work_3+work_4+work_5+work_6)/work_denom*upper_bound_work*alphas_mult
#     }

#     # Determine mixing method
#     mixing_method_before = {
#         "name":"mult",
#         "param_alpha":alpha_mixing,
#         "param_beta":beta_mixing,
#         "fixed_gamma":gamma_mixing_before,
#     }

#     # Determine mixing method
#     mixing_method_after = {
#         "name":"mult",
#         "param_alpha":alpha_mixing,
#         "param_beta":beta_mixing,
#         "fixed_gamma":gamma_mixing_after,
#     }
#     #dynModel.mixing_method = mixing_method_after

#     # Number of days
#     days_before_date_1 = int(days_ahead)
#     days_between_dates_1_2 = (date_2-date_1).days
#     days_between_dates_2_3 = (date_3-date_2).days
#     days_between_dates_3_4 = (date_4-date_3).days
#     days_between_dates_4_5 = (date_5-date_4).days
#     days_between_dates_5_6 = (date_6-date_5).days
#     days_between_dates_6_7 = (date_7-date_6).days

#     days_after_date_7 = (final_date-date_7).days
#     total_days = days_before_date_1 + days_between_dates_1_2 + days_between_dates_2_3 + days_between_dates_3_4 + days_between_dates_4_5+ days_between_dates_5_6+ days_between_dates_6_7 + days_after_date_7



#     # Calculate alphas
#     alphas_vec = []
#     for t in range(days_before_date_1):
#         alphas = {}
#         for age_group in age_groups:
#             alphas[age_group] = a_before_date_1
#         alphas_vec.append(alphas)
#     for t in range(days_between_dates_1_2):
#         alphas = {}
#         for age_group in age_groups:
#             alphas[age_group] = a_between_date_1_2
#         alphas_vec.append(alphas)
#     for t in range(days_between_dates_2_3):
#         alphas = {}
#         for age_group in age_groups:
#             alphas[age_group] = a_between_date_2_3
#         alphas_vec.append(alphas)
#     for t in range(days_between_dates_3_4):
#         alphas = {}
#         for age_group in age_groups:
#             alphas[age_group] = a_between_date_3_4
#         alphas_vec.append(alphas)
#     for t in range(days_between_dates_4_5):
#         alphas = {}
#         for age_group in age_groups:
#             alphas[age_group] = a_between_date_4_5
#         alphas_vec.append(alphas)
#     for t in range(days_between_dates_5_6):
#         alphas = {}
#         for age_group in age_groups:
#             alphas[age_group] = a_between_date_5_6
#         alphas_vec.append(alphas)
#     for t in range(days_between_dates_6_7):
#         alphas = {}
#         for age_group in age_groups:
#             alphas[age_group] = a_between_date_6_7
#         alphas_vec.append(alphas)
#     for t in range(days_after_date_7):
#         alphas = {}
#         for age_group in age_groups:
#             alphas[age_group] = a_after_date_7
#         alphas_vec.append(alphas)

#     mixing_vec = []
#     for t in range(int(gamma_change)):
#         mixing_vec.append(mixing_method_before)
#     for t in range(int(gamma_change),total_days):
#         mixing_vec.append(mixing_method_after)
        
        
        
        
#     # Calculate tests
#     tests = np.zeros(len(age_groups))


#     # Run model
#     model_data_beds = {ag:[] for ag in age_groups+["total"]}
#     model_data_icus = {ag:[] for ag in age_groups+["total"]}
#     model_data_deaths = {ag:[] for ag in age_groups+["total"]}

#     state = state_to_matrix(initialization)
#     t_beds = 0
#     t_icus = 0
#     t_deaths = 0
#     for i,ag in enumerate(age_groups):
#         state_H = state[i,cont.index("H")]
#         state_ICU = state[i,cont.index("ICU")]
#         state_D = state[i,cont.index("D")]
#         model_data_beds[ag].append(state_H)
#         model_data_icus[ag].append(state_ICU)
#         model_data_deaths[ag].append(state_D)
#         t_beds+= state_H
#         t_icus+= state_ICU
#         t_deaths+= state_D
#     model_data_beds["total"].append(t_beds)
#     model_data_icus["total"].append(t_icus)
#     model_data_deaths["total"].append(t_deaths)

#     recalc_days = [0,
#                  days_before_date_1,
#                  days_before_date_1+days_between_dates_1_2,
#                  days_before_date_1+days_between_dates_1_2+days_between_dates_2_3,
#                  days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4,
#                  days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5,
#                  days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5+days_between_dates_5_6,
#                  days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5+days_between_dates_5_6+days_between_dates_6_7,
#                 int(gamma_change)
#                 ]

#     for t in range(total_days):
#         if t in recalc_days:
#             update_contacts = True
#         else:
#             update_contacts = False
#         dynModel.mixing_method = mixing_vec[t]
#         state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), "", update_contacts=update_contacts)
#         t_beds = 0
#         t_icus = 0
#         t_deaths = 0
#         for i,ag in enumerate(age_groups):
#             state_H = state[i,cont.index("H")]
#             state_ICU = state[i,cont.index("ICU")]
#             state_D = state[i,cont.index("D")]
#             model_data_beds[ag].append(state_H)
#             model_data_icus[ag].append(state_ICU)
#             model_data_deaths[ag].append(state_D)
#             t_beds+= state_H
#             t_icus+= state_ICU
#             t_deaths+= state_D
#         model_data_beds["total"].append(t_beds)
#         model_data_icus["total"].append(t_icus)
#         model_data_deaths["total"].append(t_deaths)


#     initial_date = date_1-timedelta(days=days_before_date_1)

#     # Calculate the days of the model
#     days_model = [initial_date+timedelta(days = t) for t in range(total_days + 1)]

#     # Indices where to put the real data
#     indices = [(datetime.strptime(d, '%Y-%m-%d') - initial_date).days for d in days]

#     # Real data
#     real_data_beds = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}
#     real_data_icus = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}
#     real_data_deaths = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}

#     for k,ind in enumerate(indices):
#         for ag in age_groups+["total"]:
#             real_data_beds[ag][ind] = beds_real[ag][k]
#             real_data_icus[ag][ind] = icus_real[ag][k]
#             real_data_deaths[ag][ind] = deaths_real[ag][k]


#     error_beds = 0
#     error_icus = 0
#     error_deaths = 0
#     for ag in age_groups:
#         error_beds += np.nanmean(np.abs(np.array(model_data_beds[ag])-np.array(real_data_beds[ag])))
#         error_icus += np.nanmean(np.abs(np.array(model_data_icus[ag])-np.array(real_data_icus[ag])))
#         error_deaths += np.nanmean(np.abs(np.array(model_data_deaths[ag])-np.array(real_data_deaths[ag])))
#     error_beds_total = np.nanmean(np.abs(np.array(model_data_beds["total"])-np.array(real_data_beds["total"])))
#     error_icus_total = np.nanmean(np.abs(np.array(model_data_icus["total"])-np.array(real_data_icus["total"])))
#     error_deaths_total = np.nanmean(np.abs(np.array(model_data_deaths["total"])-np.array(real_data_deaths["total"])))

#     diff = np.array(model_data_beds["total"])-np.array(real_data_beds["total"])
#     error_beds_above = np.nanmean([max(d,0) for d in diff])
#     error_beds_below = -np.nanmean([min(d,0) for d in diff])

#     cumm_beds_model = [sum([model_data_beds["total"][k] for k in range(i+1) if not math.isnan(real_data_beds["total"][k])]) for i in range(len(model_data_beds["total"]))]
#     cumm_beds_real = [sum([real_data_beds["total"][k] for k in range(i+1) if not math.isnan(real_data_beds["total"][k])]) for i in range(len(real_data_beds["total"]))]
#     diff_cumm = np.array(cumm_beds_model)-np.array(cumm_beds_real)
#     error_cumm_above = np.nanmean([max(d,0) for d in diff_cumm])
#     error_cumm_below = -np.nanmean([min(d,0) for d in diff_cumm])



#     #     error = error_beds_total
#     #     error = mult_icus*error_icus_total
#     #     error = mult_deaths*error_deaths_total
#     upper_model_data = model_data_beds["total"]
#     upper_days_model = days_model
#     upper_real_data = real_data_beds["total"]
#     error = error_beds+5*error_beds_total

#     # plt.plot(upper_days_model, upper_model_data, label=str(days_diff))
#     # plt.plot(upper_days_model, upper_real_data, color="black")


# %%
# Modify number of days
# interval_days = [-6,6]
# plt.figure(1)
# for k in range(-4,6,2):
#     plot_model(best_v,k,1,1)
# plt.show()


# %%
# get_ipython().run_line_magic('matplotlib', 'inline')


# %%
# Modify number of days
# plt.figure(1)
# for k in np.linspace(0.95,1.05,6):
#     plot_model(best_v,0,k,1)
# plt.show()


# %%



# %%
# Modify number of days
# plt.figure(1)
# for k in np.linspace(0.97,1.03,6):
#     plot_model(best_v,0,1,k)
# plt.show()


# %%
v = best_v

days_ahead = v[0]
alpha_mixing = v[1]
beta_mixing = v[2]
gamma_mixing_before = v[3]
gamma_mixing_after = v[3]*(1-v[4])
gamma_change = v[5] + v[0]


upper_bound_home = v[6]
upper_bound_leisure = v[7]
upper_bound_other = v[8]
upper_bound_school = v[9]
upper_bound_work = v[10]
upper_bound_transport = v[11]

leisure_v = v[12:29]
leisure_1 = leisure_v[0]
leisure_2 = leisure_v[1]
leisure_3 = leisure_v[2]
leisure_4 = leisure_v[3]
leisure_5 = leisure_v[4]
leisure_6 = leisure_v[5]
leisure_7 = leisure_v[6]
leisure_denom = leisure_1+leisure_2+leisure_3+leisure_4+leisure_5+leisure_6+leisure_7

other_v = v[19:26]
other_1 = other_v[0]
other_2 = other_v[1]
other_3 = other_v[2]
other_4 = other_v[3]
other_5 = other_v[4]
other_6 = other_v[5]
other_7 = other_v[6]
other_denom = other_1+other_2+other_3+other_4+other_5+other_6+other_7

work_v = v[26:33]
work_1 = work_v[0]
work_2 = work_v[1]
work_3 = work_v[2]
work_4 = work_v[3]
work_5 = work_v[4]
work_6 = work_v[5]
work_7 = work_v[6]
work_denom = work_1+work_2+work_3+work_4+work_5+work_6+work_7

transport_v = v[33:40]
transport_1 = transport_v[0]
transport_2 = transport_v[1]
transport_3 = transport_v[2]
transport_4 = transport_v[3]
transport_5 = transport_v[4]
transport_6 = transport_v[5]
transport_7 = transport_v[6]
transport_denom = transport_1+transport_2+transport_3+transport_4+transport_5+transport_6+transport_7


# Construct initialization
initialization = copy.deepcopy(original_initialization)
for i,group in enumerate(age_groups):
    if group == "age_group_40_49":
        initialization[group]["I"] = initialization[group]["I"] + 1
        initialization[group]["S"] = initialization[group]["S"] - 1
    initialization[group]["N"] = initialization[group]["S"] + initialization[group]["E"] + initialization[group]["I"] + initialization[group]["R"]


# Alphas
a_before_date_1 = {
    'home':upper_bound_home,
    'leisure':upper_bound_leisure,
    'other':upper_bound_other,
    'school':upper_bound_school,
    'transport':upper_bound_transport,
    'work':upper_bound_work
}
a_between_date_1_2 = {
    'home':upper_bound_home,
    'leisure':leisure_1/leisure_denom*upper_bound_leisure,
    'other':other_1/other_denom*upper_bound_other,
    'school':0,
    'transport':transport_1/transport_denom*upper_bound_transport,
    'work':work_1/work_denom*upper_bound_work
}
a_between_date_2_3 = {
    'home':upper_bound_home,
    'leisure':leisure_1/leisure_denom*upper_bound_leisure,
    'other':other_1/other_denom*upper_bound_other,
    'school':0,
    'transport':transport_1/transport_denom*upper_bound_transport,
    'work':work_1/work_denom*upper_bound_work
}
a_between_date_3_4 = {
    'home':upper_bound_home,
    'leisure':(leisure_1+leisure_2)/leisure_denom*upper_bound_leisure,
    'other':(other_1+other_2)/other_denom*upper_bound_other,
    'school':0.33*upper_bound_school,
    'transport':(transport_1+transport_2)/transport_denom*upper_bound_transport,
    'work':(work_1+work_2)/work_denom*upper_bound_work
}
a_between_date_4_5 = {
    'home':upper_bound_home,
    'leisure':(leisure_1+leisure_2+leisure_3)/leisure_denom*upper_bound_leisure,
    'other':(other_1+other_2+other_3)/other_denom*upper_bound_other,
    'school':0.67*upper_bound_school,
    'transport':(transport_1+transport_2+transport_3)/transport_denom*upper_bound_transport,
    'work':(work_1+work_2+work_3)/work_denom*upper_bound_work
}
a_between_date_5_6 = {
    'home':upper_bound_home,
    'leisure':(leisure_1+leisure_2+leisure_3+leisure_4)/leisure_denom*upper_bound_leisure,
    'other':(other_1+other_2+other_3+other_4)/other_denom*upper_bound_other,
    'school':0.67*upper_bound_school,
    'transport':(transport_1+transport_2+transport_3+transport_4)/transport_denom*upper_bound_transport,
    'work':(work_1+work_2+work_3+work_4)/work_denom*upper_bound_work
}
a_between_date_6_7 = {
    'home':upper_bound_home,
    'leisure':(leisure_1+leisure_2+leisure_3+leisure_4+leisure_5)/leisure_denom*upper_bound_leisure,
    'other':(other_1+other_2+other_3+other_4+other_5)/other_denom*upper_bound_other,
    'school':1.0*upper_bound_school,
    'transport':(transport_1+transport_2+transport_3+transport_4+transport_5)/transport_denom*upper_bound_transport,
    'work':(work_1+work_2+work_3+work_4+work_5)/work_denom*upper_bound_work
}
a_after_date_7 = {
    'home':upper_bound_home,
    'leisure':(leisure_1+leisure_2+leisure_3+leisure_4+leisure_5+leisure_6)/leisure_denom*upper_bound_leisure,
    'other':(other_1+other_2+other_3+other_4+other_5+other_6)/other_denom*upper_bound_other,
    'school':1.0*upper_bound_school,
    'transport':(transport_1+transport_2+transport_3+transport_4+transport_5+transport_6)/transport_denom*upper_bound_transport,
    'work':(work_1+work_2+work_3+work_4+work_5+work_6)/work_denom*upper_bound_work
}

# Determine mixing method
mixing_method_before = {
    "name":"mult",
    "param_alpha":alpha_mixing,
    "param_beta":beta_mixing,
    "fixed_gamma":gamma_mixing_before,
}

# Determine mixing method
mixing_method_after = {
    "name":"mult",
    "param_alpha":alpha_mixing,
    "param_beta":beta_mixing,
    "fixed_gamma":gamma_mixing_after,
}
#dynModel.mixing_method = mixing_method_after

# Number of days
days_before_date_1 = int(days_ahead)
days_between_dates_1_2 = (date_2-date_1).days
days_between_dates_2_3 = (date_3-date_2).days
days_between_dates_3_4 = (date_4-date_3).days
days_between_dates_4_5 = (date_5-date_4).days
days_between_dates_5_6 = (date_6-date_5).days
days_between_dates_6_7 = (date_7-date_6).days

days_after_date_7 = (final_date-date_7).days
total_days = days_before_date_1 + days_between_dates_1_2 + days_between_dates_2_3 + days_between_dates_3_4 + days_between_dates_4_5+ days_between_dates_5_6+ days_between_dates_6_7 + days_after_date_7



# Calculate alphas
alphas_vec = []
for t in range(days_before_date_1):
    alphas = {}
    for age_group in age_groups:
        alphas[age_group] = a_before_date_1
    alphas_vec.append(alphas)
for t in range(days_between_dates_1_2):
    alphas = {}
    for age_group in age_groups:
        alphas[age_group] = a_between_date_1_2
    alphas_vec.append(alphas)
for t in range(days_between_dates_2_3):
    alphas = {}
    for age_group in age_groups:
        alphas[age_group] = a_between_date_2_3
    alphas_vec.append(alphas)
for t in range(days_between_dates_3_4):
    alphas = {}
    for age_group in age_groups:
        alphas[age_group] = a_between_date_3_4
    alphas_vec.append(alphas)
for t in range(days_between_dates_4_5):
    alphas = {}
    for age_group in age_groups:
        alphas[age_group] = a_between_date_4_5
    alphas_vec.append(alphas)
for t in range(days_between_dates_5_6):
    alphas = {}
    for age_group in age_groups:
        alphas[age_group] = a_between_date_5_6
    alphas_vec.append(alphas)
for t in range(days_between_dates_6_7):
    alphas = {}
    for age_group in age_groups:
        alphas[age_group] = a_between_date_6_7
    alphas_vec.append(alphas)
for t in range(days_after_date_7):
    alphas = {}
    for age_group in age_groups:
        alphas[age_group] = a_after_date_7
    alphas_vec.append(alphas)

mixing_vec = []
for t in range(int(gamma_change)):
    mixing_vec.append(mixing_method_before)
for t in range(int(gamma_change),total_days):
    mixing_vec.append(mixing_method_after)




# Calculate tests
tests = np.zeros(len(age_groups))


# Run model
model_data_beds = {ag:[] for ag in age_groups+["total"]}
model_data_icus = {ag:[] for ag in age_groups+["total"]}
model_data_deaths = {ag:[] for ag in age_groups+["total"]}

state = state_to_matrix(initialization)
t_beds = 0
t_icus = 0
t_deaths = 0
for i,ag in enumerate(age_groups):
    state_H = state[i,cont.index("H")]
    state_ICU = state[i,cont.index("ICU")]
    state_D = state[i,cont.index("D")]
    model_data_beds[ag].append(state_H)
    model_data_icus[ag].append(state_ICU)
    model_data_deaths[ag].append(state_D)
    t_beds+= state_H
    t_icus+= state_ICU
    t_deaths+= state_D
model_data_beds["total"].append(t_beds)
model_data_icus["total"].append(t_icus)
model_data_deaths["total"].append(t_deaths)

recalc_days = [0,
             days_before_date_1,
             days_before_date_1+days_between_dates_1_2,
             days_before_date_1+days_between_dates_1_2+days_between_dates_2_3,
             days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4,
             days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5,
             days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5+days_between_dates_5_6,
             days_before_date_1+days_between_dates_1_2+days_between_dates_2_3+days_between_dates_3_4+days_between_dates_4_5+days_between_dates_5_6+days_between_dates_6_7,
            int(gamma_change)
            ]

for t in range(total_days):
    if t in recalc_days:
        update_contacts = True
    else:
        update_contacts = False
    dynModel.mixing_method = mixing_vec[t]
    state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t, update_contacts=update_contacts)
    t_beds = 0
    t_icus = 0
    t_deaths = 0
    for i,ag in enumerate(age_groups):
        state_H = state[i,cont.index("H")]
        state_ICU = state[i,cont.index("ICU")]
        state_D = state[i,cont.index("D")]
        model_data_beds[ag].append(state_H)
        model_data_icus[ag].append(state_ICU)
        model_data_deaths[ag].append(state_D)
        t_beds+= state_H
        t_icus+= state_ICU
        t_deaths+= state_D
    model_data_beds["total"].append(t_beds)
    model_data_icus["total"].append(t_icus)
    model_data_deaths["total"].append(t_deaths)


initial_date = date_1-timedelta(days=days_before_date_1)

# Calculate the days of the model
days_model = [initial_date+timedelta(days = t) for t in range(total_days + 1)]

# Indices where to put the real data
indices = [(datetime.strptime(d, '%Y-%m-%d') - initial_date).days for d in days]

# Real data
real_data_beds = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}
real_data_icus = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}
real_data_deaths = {ag:[float('nan')]*len(days_model) for ag in age_groups+["total"]}

for k,ind in enumerate(indices):
    for ag in age_groups+["total"]:
        real_data_beds[ag][ind] = beds_real[ag][k]
        real_data_icus[ag][ind] = icus_real[ag][k]
        real_data_deaths[ag][ind] = deaths_real[ag][k]


error_beds = 0
error_icus = 0
error_deaths = 0
for ag in age_groups:
    error_beds += np.nanmean(np.abs(np.array(model_data_beds[ag])-np.array(real_data_beds[ag])))
    error_icus += np.nanmean(np.abs(np.array(model_data_icus[ag])-np.array(real_data_icus[ag])))
    error_deaths += np.nanmean(np.abs(np.array(model_data_deaths[ag])-np.array(real_data_deaths[ag])))
error_beds_total = np.nanmean(np.abs(np.array(model_data_beds["total"])-np.array(real_data_beds["total"])))
error_icus_total = np.nanmean(np.abs(np.array(model_data_icus["total"])-np.array(real_data_icus["total"])))
error_deaths_total = np.nanmean(np.abs(np.array(model_data_deaths["total"])-np.array(real_data_deaths["total"])))

diff = np.array(model_data_beds["total"])-np.array(real_data_beds["total"])
error_beds_above = np.nanmean([max(d,0) for d in diff])
error_beds_below = -np.nanmean([min(d,0) for d in diff])

cumm_beds_model = [sum([model_data_beds["total"][k] for k in range(i+1) if not math.isnan(real_data_beds["total"][k])]) for i in range(len(model_data_beds["total"]))]
cumm_beds_real = [sum([real_data_beds["total"][k] for k in range(i+1) if not math.isnan(real_data_beds["total"][k])]) for i in range(len(real_data_beds["total"]))]
diff_cumm = np.array(cumm_beds_model)-np.array(cumm_beds_real)
error_cumm_above = np.nanmean([max(d,0) for d in diff_cumm])
error_cumm_below = -np.nanmean([min(d,0) for d in diff_cumm])



#     error = error_beds_total
#     error = mult_icus*error_icus_total
#     error = mult_deaths*error_deaths_total
upper_model_data = model_data_beds["total"]
upper_days_model = days_model
upper_real_data = real_data_beds["total"]
error = error_beds+5*error_beds_total




# print(a_before_date_1)
# print(a_between_date_1_2)
# print(a_between_date_2_3)
# print(a_between_date_3_4)
# print(a_between_date_4_5)
# print(a_between_date_5_6)
# print(a_between_date_6_7)
# print(a_after_date_7)


# %%
with open("./initialization/initialization.yaml") as file:
    initialization = yaml.load(file, Loader=yaml.FullLoader)
with open("./parameters/Ile-de-France.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)


# %%
# Modify parameters
universe_params['mixing'] = {
    "name":"mult",
    "param_alpha":float(alpha_mixing),
    "param_beta":float(beta_mixing),
    "param_gamma_before":float(gamma_mixing_before),
    "param_gamma_after":float(gamma_mixing_after),
}
universe_params['days_before_gamma'] = gamma_change
universe_params['upper_bounds'] = {
    "transport":float(upper_bound_transport),
    "leisure":float(upper_bound_leisure),
    "other":float(upper_bound_other),
    "school":float(upper_bound_school),
    "home":float(upper_bound_home),
    "work":float(upper_bound_work)
}
    
with open('./parameters/fitted-alex_bkp.yaml', 'w') as file:
    yaml.dump(universe_params, file)
    


# %%
days_ahead


# %%
# Add economic value parameters
contrib = pd.read_excel("./parameter_fitting/ile-de-france_data_master.xlsx",sheet_name="contributions_normal", index_col = 0)
gains = pd.read_excel("./parameter_fitting/ile-de-france_data_master.xlsx",sheet_name="activity_levels_as_%_of_full", index_col = 0)

empl_params = {}
empl_params["v"] = {}
for age_group in age_groups:
    empl_params["v"][age_group] = {}
    for activity in econ_activities:
        empl_params["v"][age_group][activity] = float(contrib[age_group][activity])/365.0


# %%
# Calculate l-april and l-may
l_april = a_between_date_1_2
l_may = {}
for a in a_between_date_1_2.keys():
    l_may[a] = a_between_date_1_2[a]/3.0 + a_between_date_3_4[a]*2.0/3.0


# %%
l_april


# %%
l_may


# %%
eq_activities = ['leisure','other','school','transport']
t = 0.5
nu = (1-t)*(1-l_may["work"])/(1-0.5851)
eta = t*(1-np.mean([l_may[act] for act in eq_activities]))/(1-0.5851)
gamma = 1-nu-eta
# print(nu,eta,gamma)
empl_params['nu'] = 0.33
empl_params['eta'] = 0.33
empl_params['gamma'] = 0.34


# %%



# %%
# from gurobipy import *

# eq_activities = ['leisure','other','school','transport']
upper_bounds = {
    "transport":upper_bound_transport,
    "leisure":upper_bound_leisure,
    "other":upper_bound_other,
    "school":upper_bound_school,
    "home":upper_bound_home,
    "work":upper_bound_work,
}
# m = Model()
# nu = m.addVar(vtype=GRB.CONTINUOUS, name="nu", lb = 0)
# gamma = m.addVar(vtype=GRB.CONTINUOUS, name="gamma", lb = 0)
# epsilonp_1 = m.addVar(vtype=GRB.CONTINUOUS, name="epsilonp_1", lb = 0)
# epsilonn_1 = m.addVar(vtype=GRB.CONTINUOUS, name="epsilonn_1", lb = 0)
# epsilonp_2 = m.addVar(vtype=GRB.CONTINUOUS, name="epsilonp_2", lb = 0)
# epsilonn_2 = m.addVar(vtype=GRB.CONTINUOUS, name="epsilonn_2", lb = 0)

# m.addConstr(nu*upper_bound_work+gamma==1)
# m.addConstr(
#         nu*l_april["work"]  + gamma == 0.5851+epsilonp_1-epsilonn_1
# )
# m.addConstr(
#         nu*l_may["work"] + gamma == 0.7170+epsilonp_2-epsilonn_2
# )
# m.setObjective(epsilonp_1+epsilonn_1+epsilonp_2+epsilonn_2)
# m.update()
# m.optimize()
# print("nu",nu.x)
# print("gamma",gamma.x)
# empl_params['nu'] = float(nu.x)
# empl_params['gamma'] = float(gamma.x)


# %%
empl_params


# %%



# %%
# Calculate the schooling parameter
r = 0.03
schooling_param = {}
for age_group in age_groups:
    if age_group == "age_group_0_9":
        schooling_param[age_group] = (1+r)**(-15)*sum([empl_params["v"]["age_group_20_29"][act] for act in econ_activities])/upper_bound_school
    elif age_group == "age_group_10_19":
        schooling_param[age_group] = 0.907*(1+r)**(-5)*sum([empl_params["v"]["age_group_20_29"][act] for act in econ_activities])/upper_bound_school
    else:
        schooling_param[age_group] = 0


# %%
# Calculate the cost of death
econ_cost_death = {}
age_groups_n = [0,10,20,30,40,50,60,70,80]
for i,age_group in enumerate(age_groups):
    s = 0
    for tao in range(age_groups_n[i]+5,70):
        ag = "age_group_%d_%d"%(int(tao/10)*10,int(tao/10)*10+9)
        s+=(1+r)**(-(tao-age_groups_n[i]))*sum([empl_params["v"][ag][act] for act in econ_activities])*365
    econ_cost_death[age_group] = float(s)


# %%
econ_params = {
    "employment_params":empl_params,
    "schooling_params":schooling_param,
    "econ_cost_death":econ_cost_death,
    "upper_bounds":upper_bounds,
}
with open('./parameters/econ-alex_bkp.yaml', 'w') as file:
    yaml.dump(econ_params, file)


# %%
# print(econ_params)


# %%
# Initialization
# Construct initialization
initialization = copy.deepcopy(original_initialization)
for i,group in enumerate(age_groups):
    if group == "age_group_40_49":
        initialization[group]["I"] = initialization[group]["I"] + 1
        initialization[group]["S"] = initialization[group]["S"] - 1
    initialization[group]["N"] = initialization[group]["S"] + initialization[group]["E"] + initialization[group]["I"] + initialization[group]["R"]

# Run model
model_data = []
state = state_to_matrix(initialization)

for t in range(10):
    if t < universe_params['days_before_gamma']:
        lockdown_status = "pre-gamma"
    else:
        lockdown_status = "post-gamma"
    state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t)

def matrix_to_state(m):
    state = {}
    for i,age_group in enumerate(age_groups):
        state[age_group] = {}
        for j,c in enumerate(cont):
            state[age_group][c] = float(m[i,j])
    return state

with open('./initialization/10days-alex_bkp.yaml', 'w') as file:
    yaml.dump(matrix_to_state(state), file)
    


# %%
# Run model
model_data = []
state = state_to_matrix(initialization)

for t in range(20):
    if t < universe_params['days_before_gamma']:
        lockdown_status = "pre-gamma"
    else:
        lockdown_status = "post-gamma"
    state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t)


with open('./initialization/20days-alex_bkp.yaml', 'w') as file:
    yaml.dump(matrix_to_state(state), file)


# %%
# Run model
model_data = []
state = state_to_matrix(initialization)

for t in range(30):
    if t < universe_params['days_before_gamma']:
        lockdown_status = "pre-gamma"
    else:
        lockdown_status = "post-gamma"
    state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t)

with open('./initialization/30days-alex_bkp.yaml', 'w') as file:
    yaml.dump(matrix_to_state(state), file)


# %%
# Run model
model_data = []
state = state_to_matrix(initialization)

for t in range(40):
    if t < universe_params['days_before_gamma']:
        lockdown_status = "pre-gamma"
    else:
        lockdown_status = "post-gamma"
    state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t)

with open('./initialization/40days-alex_bkp.yaml', 'w') as file:
    yaml.dump(matrix_to_state(state), file)


# %%
# Run model
model_data = []
state = state_to_matrix(initialization)

for t in range(50):
    if t < universe_params['days_before_gamma']:
        lockdown_status = "pre-gamma"
    else:
        lockdown_status = "post-gamma"
    state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t)


with open('./initialization/50days-alex_bkp.yaml', 'w') as file:
    yaml.dump(matrix_to_state(state), file)


# %%
# Run model
model_data = []
state = state_to_matrix(initialization)

for t in range(60):
    if t < universe_params['days_before_gamma']:
        lockdown_status = "pre-gamma"
    else:
        lockdown_status = "post-gamma"
    state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t)


with open('./initialization/60days-alex_bkp.yaml', 'w') as file:
    yaml.dump(matrix_to_state(state), file)

objects = []
with open('./initialization/alldays-alex_bkp.yaml', 'w') as file:
    for t in range((final_date-date_1).days):
        state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t)
        current_date = date_1 + timedelta(days=t + 1)
        obj = {}
        obj["date"] = current_date.strftime("%Y-%m-%d")
        obj_state = matrix_to_state(state)
        for obj_state_key in obj_state:
            obj[obj_state_key] = obj_state[obj_state_key]
        objects.append(obj)
        
    # for obj in objects:
    yaml.dump(objects, file)


# %%
a_before_date_1 = {
    'home':float(upper_bound_home),
    'leisure':float(upper_bound_leisure),
    'other':float(upper_bound_other),
    'school':float(upper_bound_school),
    'transport':float(upper_bound_transport),
    'work':float(upper_bound_work),
}
a_between_date_1_2 = {
    'home':float(upper_bound_home),
    'leisure':float(leisure_1/leisure_denom*upper_bound_leisure),
    'other':float(other_1/other_denom*upper_bound_other),
    'school':0,
    'transport':float(transport_1/transport_denom*upper_bound_transport),
    'work':float(work_1/work_denom*upper_bound_work),
}
a_between_date_2_3 = {
    'home':float(upper_bound_home),
    'leisure':float(leisure_1/leisure_denom*upper_bound_leisure),
    'other':float(other_1/other_denom*upper_bound_other),
    'school':0,
    'transport':float(transport_1/transport_denom*upper_bound_transport),
    'work':float(work_1/work_denom*upper_bound_work),
}
a_between_date_3_4 = {
    'home':float(upper_bound_home),
    'leisure':float((leisure_1+leisure_2)/leisure_denom*upper_bound_leisure),
    'other':float((other_1+other_2)/other_denom*upper_bound_other),
    'school':float(0.33*upper_bound_school),
    'transport':float((transport_1+transport_2)/transport_denom*upper_bound_transport),
    'work':float((work_1+work_2)/work_denom*upper_bound_work),
}
a_between_date_4_5 = {
    'home':float(upper_bound_home),
    'leisure':float((leisure_1+leisure_2+leisure_3)/leisure_denom*upper_bound_leisure),
    'other':float((other_1+other_2+other_3)/other_denom*upper_bound_other),
    'school':float(0.67*upper_bound_school),
    'transport':float((transport_1+transport_2+transport_3)/transport_denom*upper_bound_transport),
    'work':float((work_1+work_2+work_3)/work_denom*upper_bound_work),
}
a_between_date_5_6 = {
    'home':float(upper_bound_home),
    'leisure':float((leisure_1+leisure_2+leisure_3+leisure_4)/leisure_denom*upper_bound_leisure),
    'other':float((other_1+other_2+other_3+other_4)/other_denom*upper_bound_other),
    'school':float(0.67*upper_bound_school),
    'transport':float((transport_1+transport_2+transport_3+transport_4)/transport_denom*upper_bound_transport),
    'work':float((work_1+work_2+work_3+work_4)/work_denom*upper_bound_work),
}
a_between_date_6_7 = {
    'home':float(upper_bound_home),
    'leisure':float((leisure_1+leisure_2+leisure_3+leisure_4+leisure_5)/leisure_denom*upper_bound_leisure),
    'other':float((other_1+other_2+other_3+other_4+other_5)/other_denom*upper_bound_other),
    'school':float(1.0*upper_bound_school),
    'transport':float((transport_1+transport_2+transport_3+transport_4+transport_5)/transport_denom*upper_bound_transport),
    'work':float((work_1+work_2+work_3+work_4+work_5)/work_denom*upper_bound_work),
}
a_after_date_7 = {
    'home':float(upper_bound_home),
    'leisure':float((leisure_1+leisure_2+leisure_3+leisure_4+leisure_5+leisure_6)/leisure_denom*upper_bound_leisure),
    'other':float((other_1+other_2+other_3+other_4+other_5+other_6)/other_denom*upper_bound_other),
    'school':float(1.0*upper_bound_school),
    'transport':float((transport_1+transport_2+transport_3+transport_4+transport_5+transport_6)/transport_denom*upper_bound_transport),
    'work':float((work_1+work_2+work_3+work_4+work_5+work_6)/work_denom*upper_bound_work),
}


# %%
# Construct alphas
from copy import deepcopy
alphas = []

c = 0
# Alphas
for i in range(days_before_date_1):
    a_before_date_1.update({
        'date':(initial_date + timedelta(days=c)).strftime('%Y-%m-%d'),
        'days_from_lockdown':c-days_before_date_1,
    })
    alphas.append(deepcopy(a_before_date_1))
    c+=1
for i in range(days_between_dates_1_2):
    a_between_date_1_2.update({
        'date':(initial_date + timedelta(days=c)).strftime('%Y-%m-%d'),
        'days_from_lockdown':c-days_before_date_1,
    })
    alphas.append(deepcopy(a_between_date_1_2))
    c+=1
for i in range(days_between_dates_2_3):
    a_between_date_2_3.update({
        'date':(initial_date + timedelta(days=c)).strftime('%Y-%m-%d'),
        'days_from_lockdown':c-days_before_date_1,
    })
    alphas.append(deepcopy(a_between_date_2_3))
    c+=1
for i in range(days_between_dates_3_4):
    a_between_date_3_4.update({
        'date':(initial_date + timedelta(days=c)).strftime('%Y-%m-%d'),
        'days_from_lockdown':c-days_before_date_1,
    })
    alphas.append(deepcopy(a_between_date_3_4))
    c+=1
for i in range(days_between_dates_4_5):
    a_between_date_4_5.update({
        'date':(initial_date + timedelta(days=c)).strftime('%Y-%m-%d'),
        'days_from_lockdown':c-days_before_date_1,
    })
    alphas.append(deepcopy(a_between_date_4_5))
    c+=1
for i in range(days_between_dates_5_6):
    a_between_date_5_6.update({
        'date':(initial_date + timedelta(days=c)).strftime('%Y-%m-%d'),
        'days_from_lockdown':c-days_before_date_1,
    })
    alphas.append(deepcopy(a_between_date_5_6))
    c+=1
for i in range(days_between_dates_6_7):
    a_between_date_6_7.update({
        'date':(initial_date + timedelta(days=c)).strftime('%Y-%m-%d'),
        'days_from_lockdown':c-days_before_date_1,
    })
    alphas.append(deepcopy(a_between_date_6_7))
    c+=1
for i in range(days_after_date_7):
    a_after_date_7.update({
        'date':(initial_date + timedelta(days=c)).strftime('%Y-%m-%d'),
        'days_from_lockdown':c-days_before_date_1,
    })
    alphas.append(deepcopy(a_after_date_7))
    c+=1

with open('./policies/fitted-alex_bkp.yaml', 'w') as file:
    yaml.dump(alphas, file)


# %%
del a_between_date_1_2['date']
del a_between_date_1_2['days_from_lockdown']
# print(a_between_date_1_2)


# %%
with open('./lower_bounds/fitted-alex_bkp.yaml', 'w') as file:
    yaml.dump(a_between_date_1_2, file)


# %%


