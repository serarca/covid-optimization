import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from datetime import datetime, timedelta
import pandas as pd

from time import time


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
import random


from fast_group import FastDynamicalModel
from aux import *

from copy import deepcopy
import copy
import math
import matplotlib.dates as mdates

# We define the way that parameter fluctuations will be 


data = pd.read_csv("donnees-hospitalieres-classe-age-covid19-2020-10-28-19h00.csv", sep=";")


google = pd.read_csv("2020_FR_Region_Mobility_Report.csv")
google = google[google['sub_region_1']=="Île-de-France"]
google = google[pd.isnull(google['sub_region_2'])]

# Apply an smoothing to the data
initial_work = ((google["workplaces_percent_change_from_baseline"]+100)/100).values
google_days = [datetime.strptime(google["date"].values[i], '%Y-%m-%d').weekday() for i in range(len(google["date"]))]
new_work = []
for i in range(len(initial_work)):
    if google_days[i]<=4:
        new_work.append(initial_work[i])
    elif google_days[i]==5:
        numbers_for_mean = []
        for k in range(i-5,i):
            if k>=0 and k<len(initial_work):
                numbers_for_mean.append(initial_work[k])
        for k in range(i+2,i+7):
            if k>=0 and k<len(initial_work):
                numbers_for_mean.append(initial_work[k])
        new_work.append(np.mean(numbers_for_mean))
    elif google_days[i]==6:
        numbers_for_mean = []
        for k in range(i-6,i-1):
            if k>=0 and k<len(initial_work):
                numbers_for_mean.append(initial_work[k])
        for k in range(i+1,i+6):
            if k>=0 and k<len(initial_work):
                numbers_for_mean.append(initial_work[k])
        new_work.append(np.mean(numbers_for_mean))


# In[6]:


# Assign data
google['work'] = new_work
google['transport'] = (google["transit_stations_percent_change_from_baseline"]+100)/100
google['other'] = 0.33*(google["retail_and_recreation_percent_change_from_baseline"]+100)/100+0.67*(google["grocery_and_pharmacy_percent_change_from_baseline"]+100)/100
google['leisure'] = 0.33*(google["parks_percent_change_from_baseline"]+100)/100+0.67*(google["retail_and_recreation_percent_change_from_baseline"]+100)/100
google = google.reset_index()
google.to_csv('smoothed.csv')




# Load the original SEIR parameters
mean_params = pd.read_excel("./ile-de-france_data_master.xlsx",sheet_name="SEIR_params", index_col = 0, engine='openpyxl')

lower_bound_params = pd.read_excel("./ile-de-france_data_master.xlsx",sheet_name="SEIR_params_conf_range_lower", index_col = 0, engine='openpyxl')

upper_bound_params = pd.read_excel("./ile-de-france_data_master.xlsx",sheet_name="SEIR_params_conf_range_upper", index_col = 0, engine='openpyxl')


lower_r0 = 2.8
upper_r0 = 3
lower_beta = 2.8 * (0.03923794546657925/2.9)
upper_beta = 3 * (0.03923794546657925/2.9)
mid_beta = 0.03923794546657925
mid_r0 = (upper_r0 + lower_r0)/2

mean_init_param = {
    "beta":mid_beta,
    "p_ss":mean_params['p_ss'].values,
}

sigma_init_param = {
    "beta": (upper_beta - lower_beta) / 4,
    "p_ss":((upper_bound_params['p_ss'] - lower_bound_params['p_ss'])/4).values
}

params = pd.read_excel("./ile-de-france_data_master.xlsx",sheet_name="SEIR_params", index_col = 0, engine='openpyxl')





start_time = time()
for sim_i in range(0, 125):

    np.random.seed(sim_i)

    instance_variation_beta_z = np.random.randn()
    instance_variation_p_ss_z = np.random.randn()
    instance_variation_alpha_z = np.random.randn()



    age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29','age_group_30_39', 'age_group_40_49', 'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
    old_age_groups = ['age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
    econ_activities = ['transport', 'leisure', 'other']
    cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips',            'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]



    initial_params = {
        "mu":params['mu'].values,
        "sigma":params['sigma'].values,
        "p_ICU":params['p_ICU'].values[:9],
        "p_H":params['p_H'].values[:9],
        "lambda_H_R":params['lambda_HR'].values[:9],
        "lambda_H_D":params['lambda_HD'].values[:9],
        "lambda_ICU_R":params['lambda_ICUR'].values[:9],
        "lambda_ICU_D":params['lambda_ICUD'].values[:9],
        "lambda_ICU":params['lambda_ICU'].values[:9],
        "lambda_H":params['lambda_H'].values[:9],
        "p_death":params['p_death_cond_ss'].values[:9],
        "p_recov":params['p_recov_cond_ss'].values[:9],
        
    }
    
    # print(mean_init_param["p_ss"][:9])
    # print(lower_bound_params['p_ss'].to_list())
    # print(mean_init_param["p_ss"][:9] + instance_variation_p_ss_z * sigma_init_param["p_ss"][:9])

    # print(mean_init_param["p_ss"])
    # print(lower_bound_params['p_ss'].to_list())

    # print(np.minimum(np.maximum(mean_init_param["p_ss"][:9]+ instance_variation_p_ss_z * sigma_init_param["p_ss"], lower_bound_params['p_ss'].to_list()), upper_bound_params['p_ss'].to_list()))



    initial_params["p_ss"] = np.minimum(np.maximum(mean_init_param["p_ss"][:9]+ instance_variation_p_ss_z * sigma_init_param["p_ss"][:9], lower_bound_params['p_ss'].to_list()[:9]), upper_bound_params['p_ss'].to_list()[:9])

    # print(np.minimum(np.maximum(mean_init_param["beta"] + instance_variation_beta_z * sigma_init_param["beta"], lower_beta), upper_beta))

    
    print(float(mean_init_param["beta"] + instance_variation_beta_z * sigma_init_param["beta"]))
    print(lower_beta)
    print(upper_beta)

    initial_params["beta"] = float(np.minimum(np.maximum(mean_init_param["beta"] + instance_variation_beta_z * sigma_init_param["beta"], lower_beta), upper_beta))

    initial_params["p_H"] = (initial_params["p_ss"][:9] * (1/(1 + initial_params["p_ICU"][:9]/initial_params["p_H"][:9])))[:9]

    initial_params["p_ICU"] = initial_params["p_ss"][:9] - initial_params["p_H"][:9]

    # initial_params["p_ss"] = list(initial_params["p_ss"])[:9]
    # initial_params["p_H"] = list(initial_params["p_H"])[:9]
    # initial_params["p_ICU"] = list(initial_params["p_ICU"])[:9]

    # print(initial_params["p_ss"])
    # print(initial_params["p_H"])
    # print(initial_params["p_ICU"])

    # Estimated Alpha Mixing is 0.39 see table S5
    alpha_mixing = 0.39 + instance_variation_alpha_z * (0.7-0.1)/4




    region = "Ile-de-France"

    # Read group parameters
    with open("../parameters/"+region+".yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        universe_params = yaml.load(file, Loader=yaml.FullLoader)
        
    # Read initialization
    with open("../initialization/patient_zero.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        original_initialization = yaml.load(file, Loader=yaml.FullLoader)

    # Read econ parameters
    with open("../parameters/econ.yaml") as file:
        econ_params = yaml.load(file, Loader=yaml.FullLoader)

    experiment_params = {
        'delta_schooling':0.5,
        'xi':0,
        'icus':4000,
    }


    # Create relevant dates
    date_1 = datetime.strptime("2020-03-17", '%Y-%m-%d')
    date_2 = datetime.strptime("2020-05-11", '%Y-%m-%d')
    date_3 = datetime.strptime("2020-07-01", '%Y-%m-%d')
    date_4 = datetime.strptime("2020-09-01", '%Y-%m-%d')
    final_date = datetime.strptime("2020-10-21", '%Y-%m-%d')

    rates_date = datetime.strptime("2020-03-01", '%Y-%m-%d')

    first_day_google = datetime.strptime("2020-02-15", '%Y-%m-%d')
    days_between_google = (date_1-first_day_google).days

    # Fill up the original parameters
    original_beta = initial_params['beta']
    t_days_beta = 730

    

    for ag in universe_params['seir-groups']:
        universe_params['seir-groups'][ag]["parameters"]["beta"] = (
            [original_beta for t in range(t_days_beta)]
        )



    # Create model
    mixing_method = {}
    dynModel = FastDynamicalModel(universe_params, econ_params, experiment_params, 1, mixing_method, 1e9, 0, 0)


    best_v = 0
    best_error = float('inf')
    validation_date = datetime.strptime("2020-10-21", '%Y-%m-%d')
    # This is the smoothening
    trans_days = 15

    #Days Ahead computed as March 17th - Jan 1st
    days_ahead = 76

   
    # Estimated beta is 0.03923794546657925 from Ile-de France data 
    beta_normal = original_beta
    
    #Beta Multiplier is 0.6 from Table S5
    beta_masks = beta_normal*(0.6)

    # Alpha-other is 0.96 see Table S5
    mix_1 = 0.96
    

    alphas_d = {
        'work':alpha_mixing,
        'transport':alpha_mixing,
        'school':alpha_mixing,
        'other':alpha_mixing,
        'leisure':alpha_mixing,
        'home':alpha_mixing,
    }

    # School lockdowns
    
    # School Activities from S5
    school_lockdown = 0
    school_may_jun = 0
    school_jul_aug = 0.2
    school_sep_oct = 1

    # Changes to SEIR rates
    #Mult p ICU
    fraction_p_ICU = 1.66
    # Mult lambda H
    change_lambda_H_aft = 1.5
    # Mult lambda ICU
    change_lambda_ICU_aft = 2
    # Mult P Death
    change_p_death_aft = 1.13

    # Date that a change occurs
    # April 17th - March 17th See Table S5
    days_change_rates = 31


    google['other'] = mix_1*(google["retail_and_recreation_percent_change_from_baseline"]+100)/100+(1-mix_1)*(google["grocery_and_pharmacy_percent_change_from_baseline"]+100)/100
    google['leisure'] = (google["retail_and_recreation_percent_change_from_baseline"] + 100)/100


    # Number of days
    days_before_date_1 = int(days_ahead)
    days_between_dates_1_2 = (date_2-date_1).days
    days_between_dates_2_3 = (date_3-date_2).days
    days_between_dates_3_4 = (date_4-date_3).days
    days_after_date_4 = (final_date-date_4).days
    total_days = days_before_date_1 + days_between_dates_1_2 + days_between_dates_2_3 + days_between_dates_3_4 + days_after_date_4
    validation_days = days_before_date_1 + (final_date-date_1).days

    # Some additional calculations
    days_rates_start = (rates_date - date_1).days + days_ahead

    # Construct initialization
    initialization = deepcopy(original_initialization)
    for i,group in enumerate(age_groups):
        if group == "age_group_40_49":
            initialization[group]["I"] = initialization[group]["I"] + 1
            initialization[group]["S"] = initialization[group]["S"] - 1
        initialization[group]["N"] = initialization[group]["S"] + initialization[group]["E"] + initialization[group]["I"] + initialization[group]["R"]


    # Alphas
    a_before_google = {
        'home':1.0,
        'leisure':1.0,
        'other':1.0,
        'school':1.0,
        'transport':1.0,
        'work':1.0
    }



    # Calculate alphas
    alphas_vec = []
    for t in range(days_before_date_1-days_between_google):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = a_before_google
        alphas_vec.append(alphas)

    counter = 0
    for t in range(days_between_google):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = {
                'home':1.0,
                'leisure':google['leisure'][counter],
                'other':google['other'][counter],
                'school':1.0,
                'transport':google['transport'][counter],
                'work':google['work'][counter],
            }
        alphas_vec.append(alphas)   
        counter += 1

    for t in range(days_between_dates_1_2):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = {
                'home':1.0,
                'leisure':google['leisure'][counter],
                'other':google['other'][counter],
                'school':school_lockdown,
                'transport':google['transport'][counter],
                'work':google['work'][counter],
            }
        alphas_vec.append(alphas)   
        counter += 1

    for t in range(days_between_dates_2_3):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = {
                'home':1.0,
                'leisure':google['leisure'][counter],
                'other':google['other'][counter],
                'school':school_may_jun,
                'transport':google['transport'][counter],
                'work':google['work'][counter],
            }
        alphas_vec.append(alphas)   
        counter += 1

    for t in range(days_between_dates_3_4):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = {
                'home':1.0,
                'leisure':google['leisure'][counter],
                'other':google['other'][counter],
                'school':school_jul_aug,
                'transport':google['transport'][counter],
                'work':google['work'][counter],
            }
        alphas_vec.append(alphas)   
        counter += 1

    for t in range(days_after_date_4):
        alphas = {}
        for age_group in age_groups:
            alphas[age_group] = {
                'home':1.0,
                'leisure':google['leisure'][counter],
                'other':google['other'][counter],
                'school':school_sep_oct,
                'transport':google['transport'][counter],
                'work':google['work'][counter],
            }
        alphas_vec.append(alphas)
        counter += 1

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


    dynModel.beta = np.zeros((len(age_groups),len(dynModel.groups[age_groups[0]].parameters["beta"])))
    for i in range(len(age_groups)):
        for j in range(len(dynModel.groups[age_groups[0]].parameters["beta"])):
            if j < int(days_rates_start+days_change_rates):
                dynModel.beta[i,j] = beta_normal
            else:
                d_after = j-int(days_rates_start+days_change_rates)
                d_portion = max(trans_days-d_after,0)/trans_days
                dynModel.beta[i,j] = beta_normal*d_portion+beta_masks*(1-d_portion)

    for t in range(total_days):
        current_date = date_1+timedelta(days=t-days_before_date_1)
        day_of_week = current_date.weekday()

        update_contacts = True

        if t<int(days_rates_start+days_change_rates):
            dynModel.p_H = initial_params["p_H"]
            dynModel.p_ICU = initial_params["p_ICU"]

            lambda_H = initial_params["lambda_H"] 
            lambda_ICU = initial_params["lambda_ICU"]
            p_death = initial_params["p_death"]

            dynModel.lambda_H_D = p_death*lambda_H
            dynModel.lambda_H_R = (1-p_death)*lambda_H
            dynModel.lambda_ICU_D = p_death*lambda_ICU
            dynModel.lambda_ICU_R = (1-p_death)*lambda_ICU

        else:
            d_after = t-int(days_rates_start+days_change_rates)
            d_portion = max(trans_days-d_after,0)/trans_days

            dynModel.p_ICU = initial_params["p_ICU"]*d_portion+initial_params["p_ICU"]*fraction_p_ICU*(1-d_portion)
            dynModel.p_H = (initial_params["p_H"]+initial_params["p_ICU"])-(initial_params["p_ICU"]*d_portion+initial_params["p_ICU"]*fraction_p_ICU*(1-d_portion))

            lambda_H = initial_params["lambda_H"]*d_portion+initial_params["lambda_H"]*change_lambda_H_aft*(1-d_portion)
            lambda_ICU = initial_params["lambda_ICU"]*d_portion+ initial_params["lambda_ICU"]*change_lambda_ICU_aft*(1-d_portion)
            p_death = initial_params["p_death"]*d_portion+initial_params["p_death"]*change_p_death_aft*(1-d_portion)

            dynModel.lambda_H_D = p_death*lambda_H
            dynModel.lambda_H_R = (1-p_death)*lambda_H
            dynModel.lambda_ICU_D = p_death*lambda_ICU
            dynModel.lambda_ICU_R = (1-p_death)*lambda_ICU




        season = "spc"
        dynModel.mixing_method = {
                "name":"mult",
                "param_alpha":alphas_d,
                "param_beta":alphas_d,
            }

        state,_ = dynModel.take_time_step(state, tests, tests, alphas_to_matrix(alphas_vec[t]), t, season, update_contacts=update_contacts)
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
        #print(t_beds)
        model_data_beds["total"].append(t_beds)
        model_data_icus["total"].append(t_icus)
        model_data_deaths["total"].append(t_deaths)
        
    final_state = state


    initial_date = date_1-timedelta(days=days_before_date_1)

    # Calculate the days of the model
    days_model = [initial_date+timedelta(days = t) for t in range(total_days + 1)]


    upper_days_model = days_model

    # error = error_beds_total+error_beds_tail+error_beds_peak+error_icus_total+error_icus_tail+error_icus_peak+error_deaths_total+error_deaths_tail+error_deaths_peak

    # print("Error:", error)

    # In[8]:





    with open("../parameters/Ile-de-France.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        universe_params = yaml.load(file, Loader=yaml.FullLoader)

    # Modify parameters
    universe_params['mixing'] = {
        "name":"mult",
        "param_alpha":float(alpha_mixing),
        "param_beta":float(alpha_mixing),
    }
    universe_params['upper_bounds'] = {
        "transport":1.0,
        "leisure":1.0,
        "other":1.0,
        "school":1.0,
        "home":1.0,
        "work":1.0
    }
    # Modify beta
    for ag in universe_params['seir-groups']:
        universe_params['seir-groups'][ag]['parameters']['beta'] = [beta_masks for t in range(720)]

    new_p_ICU = initial_params["p_ICU"]*fraction_p_ICU
    new_p_H = (initial_params["p_H"]+initial_params["p_ICU"])-(initial_params["p_ICU"]*fraction_p_ICU)

    new_lambda_H = initial_params["lambda_H"]*change_lambda_H_aft
    new_lambda_ICU = initial_params["lambda_ICU"]*change_lambda_ICU_aft
    new_p_death = initial_params["p_death"]*change_p_death_aft

    new_lambda_H_D = new_p_death*new_lambda_H
    new_lambda_H_R = (1-new_p_death)*new_lambda_H
    new_lambda_ICU_D = new_p_death*new_lambda_ICU
    new_lambda_ICU_R = (1-new_p_death)*new_lambda_ICU


    for i,ag in enumerate(age_groups):
        universe_params['seir-groups'][ag]['parameters']['p_ICU'] = float(new_p_ICU[i])
        universe_params['seir-groups'][ag]['parameters']['p_H'] = float(new_p_H[i])
        universe_params['seir-groups'][ag]['parameters']['lambda_H_D'] = float(new_lambda_H_D[i])
        universe_params['seir-groups'][ag]['parameters']['lambda_H_R'] = float(new_lambda_H_R[i])
        universe_params['seir-groups'][ag]['parameters']['lambda_ICU_D'] = float(new_lambda_ICU_D[i])
        universe_params['seir-groups'][ag]['parameters']['lambda_ICU_R'] = float(new_lambda_ICU_R[i])
        del universe_params['seir-groups'][ag]['parameters']['p_ICU_cond_ss']
        del universe_params['seir-groups'][ag]['parameters']['p_death_cond_ss']
        
    with open('../parameters/multiple_initial_conditions/fitted_%d.yaml'%(sim_i), 'w') as file:
        yaml.dump(universe_params, file, encoding='utf-8', allow_unicode=True)
        
        


    # In[10]:


    # Add economic value parameters
    contrib = pd.read_excel("./ile-de-france_data_master.xlsx",sheet_name="contributions_normal", index_col = 0, engine='openpyxl')
    gains = pd.read_excel("./ile-de-france_data_master.xlsx",sheet_name="activity_levels_as_%_of_full", index_col = 0,engine='openpyxl')

    empl_params = {}
    empl_params["v"] = {}
    for age_group in age_groups:
        empl_params["v"][age_group] = {}
        for activity in econ_activities:
            empl_params["v"][age_group][activity] = float(contrib[age_group][activity])/365.0


    # Calculate l-april and l-may
    l_april = np.mean([google['work'][i] for i in range(len(google)) if google['date'][i][0:7]=="2020-04"])
    # print('l_april',l_april)
    l_may = np.mean([google['work'][i] for i in range(len(google)) if google['date'][i][0:7]=="2020-05"])
    # print("april",l_april)
    # print("may",l_may)


    eq_activities = ['leisure','other','school','transport']



    nu = (1-0.5851) / (1-l_april)
    gamma = 1-nu

    empl_params['nu'] = 0.95*float(nu)
    empl_params['gamma'] = float(gamma)
    empl_params['eta'] = 0.05*float(nu)

    # print("nu",empl_params['nu'])
    # print("eta",empl_params['eta'])
    # print("gamma",empl_params['gamma'])



    # Calculate the schooling parameter
    r = 0.03
    schooling_param = {}
    for age_group in age_groups:
        if age_group == "age_group_0_9":
            schooling_param[age_group] = (1+r)**(-15)*sum([empl_params["v"]["age_group_20_29"][act] for act in econ_activities])
        elif age_group == "age_group_10_19":
            schooling_param[age_group] = 0.907*(1+r)**(-5)*sum([empl_params["v"]["age_group_20_29"][act] for act in econ_activities])
        else:
            schooling_param[age_group] = 0
            

    # Calculate the cost of death
    econ_cost_death = {}
    age_groups_n = [0,10,20,30,40,50,60,70,80]
    for i,age_group in enumerate(age_groups):
        s = 0
        for tao in range(age_groups_n[i]+5,70):
            ag = "age_group_%d_%d"%(int(tao/10)*10,int(tao/10)*10+9)
            s+=(1+r)**(-(tao-age_groups_n[i]))*sum([empl_params["v"][ag][act] for act in econ_activities])*365
        econ_cost_death[age_group] = float(s)


    # In[13]:


    upper_bounds = {
        "transport":1.0,
        "leisure":1.0,
        "other":1.0,
        "school":1.0,
        "home":1.0,
        "work":1.0
    }

    econ_params = {
        "employment_params":empl_params,
        "schooling_params":schooling_param,
        "econ_cost_death":econ_cost_death,
        "upper_bounds":upper_bounds,
    }
    with open('../parameters/multiple_initial_conditions/econ_%d.yaml'%(sim_i), 'w') as file:
        yaml.dump(econ_params, file, encoding='utf-8', allow_unicode=True)


    # In[14]:


    def matrix_to_state(m):
        state = {}
        for i,age_group in enumerate(age_groups):
            state[age_group] = {}
            for j,c in enumerate(cont):
                state[age_group][c] = float(m[i,j])
        return state

    with open('../parameters/multiple_initial_conditions/oct21_%d.yaml'%(sim_i), 'w') as file:
        yaml.dump(matrix_to_state(final_state), file, encoding='utf-8', allow_unicode=True)


end_time = time()

print(f"total time: {end_time - start_time}")




# %%
