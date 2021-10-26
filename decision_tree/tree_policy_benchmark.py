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
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, current_dir+"/heuristics")
sys.path.insert(0, parent_dir+"/fast_gradient")
sys.path.insert(0, current_dir+"/heuristics/LP-Models")
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from group import SEIR_group, DynamicalModel
import math
import pprint
import pandas as pd
import pickle
import numpy as np

from scipy.optimize import Bounds,minimize,LinearConstraint
import sklearn.metrics as metrics
import sklearn
from sklearn import tree
assert sklearn.__version__ >= "0.20"  # Scikit-Learn version ≥0.20 required

def number_to_age_group(age_number):
    if age_number == 5:
        return 'age_group_0_9'
    if age_number == 15:
        return 'age_group_10_19'
    if age_number == 25:
        return 'age_group_20_29'
    if age_number == 35:
        return 'age_group_30_39'
    if age_number == 45:
        return 'age_group_40_49'
    if age_number == 55:
        return 'age_group_50_59'
    if age_number == 65:
        return 'age_group_60_69'
    if age_number == 75:
        return 'age_group_70_79'
    if age_number == 85:
        return 'age_group_80_plus'
    return "error_wrong_number_for_age"

def get_feature_set_from_name (feature_set_name):
    
    simple_features = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_D', 'econ_gradient_normalized',  'R_perc', 'ICU_utilization',
    'contacts_received', 'contacts_given', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'age']

    original_plus = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_H', 'p_ICU', 'p_D', 'econ_gradient_normalized', 'I_perc', 'R_perc', 'H_perc', 'ICU_utilization',
    'contacts_received', 'contacts_received_into_deaths', 'infections_received', 'deaths_received', 'deaths_received_econ',
    'contacts_given', 'contacts_given_into_deaths', 'infections_given', 'deaths_given', 'deaths_given_econ',
    'econ_contacts_ratio', 'age'] 

    original = ['t', 'icu_capacity', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_H', 'p_ICU', 'p_D', 'econ_gradient_normalized', 'I_perc', 'R_perc', 'H_perc', 'ICU_utilization',
    'contacts_received', 'contacts_received_into_deaths', 'infections_received', 'deaths_received', 'deaths_received_econ',
    'contacts_given', 'contacts_given_into_deaths', 'infections_given', 'deaths_given', 'deaths_given_econ',
    'econ_contacts_ratio', 'age']

    engineered_econ_to_contacts_ratio_plus = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_D', 'R_perc',  'ICU_utilization', 
    'econ_contacts_ratio', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'age']

    engineered_econ_to_contacts_ratio_minus = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_D', 'R_perc',  'ICU_utilization', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'age']

 # engineered_1 = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    # 'chi+vg_life', 'p_D', 'econ_gradient_normalized', 'R_perc',  'ICU_utilization',
    # 'contacts_received', 'contacts_received_into_deaths', 'infections_received', 'deaths_received', 'deaths_received_econ',
    # 'contacts_given', 'contacts_given_into_deaths', 'infections_given', 'deaths_given', 'deaths_given_econ',
    # 'econ_contacts_ratio', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    # 'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'age']

    # features  for the simple tree and econ to contacts ratio
    # simple_plus_features = simple_features + ['econ_contacts_ratio']
    simple_plus_features = simple_features + ['econ_contacts_ratio']

    
    if feature_set_name == "ORIGINAL":
        return original
    if feature_set_name == "SIMPLE":
        return simple_features
    if feature_set_name == "ORIGINAL_PLUS":
        return original_plus
    
    if feature_set_name == "SIMPLE_PLUS":
        return simple_plus_features

    if feature_set_name == "ENGINEERED_1_PLUS":
        return engineered_econ_to_contacts_ratio_plus
    
    if feature_set_name == "ENGINEERED_1_MINUS":
        return engineered_econ_to_contacts_ratio_minus
    
    return f"error: {feature_set_name} not a feature set name"

def get_features(dynModel, state, incidence_rate, incidence_rate_old, sevendays_avg_newICU, newICU, t, tree_file):
    
    print(f"getting feaures time {t}")
    feature_set_name = tree_file.split('_variables_')[1]

    feature_set = get_feature_set_from_name(feature_set_name)
   
    
    
    
    age_groups = dynModel.age_groups
    
    dynModel.calculate_econ_gradients(state)
    
    rel_activities = ['leisure','other','school','transport','work']

    features = {}
    for act in rel_activities:
        features[act] = []
    ## TODO: 
    ##        - write the filter for the features
    ##        - Write different versions of the huer for the diff features 

    for act in rel_activities:
        for ag in age_groups:

            GDP_per_capita = 37199.03

            all_splits = ag.split("_")
            lower_lim = np.int(all_splits[-2])
            upper_lim = all_splits[-1]
            if upper_lim=="plus":
                upper_lim = 89
            else:
                upper_lim = np.int(upper_lim)



            p_death = 1 / (dynModel.parameters["seir-groups"][ag]['parameters']['lambda_H_R'] / dynModel.parameters["seir-groups"][ag]['parameters']['lambda_H_D'] + 1)

           
            features[act].append({"age": (lower_lim + upper_lim + 1)/2,
						"t":t,
						"icu_capacity":dynModel.experiment_params["icus"],
						"theta_schooling": dynModel.experiment_params["delta_schooling"],
						"beta": dynModel.parameters["seir-groups"][ag]['parameters']['beta'][0],
						"alpha_1_2": dynModel.parameters["mixing"]['param_alpha'],
						"nu_other_act":dynModel.econ_params["employment_params"]["eta"],
						"nu_work":dynModel.econ_params["employment_params"]['nu'],
						"chi":dynModel.experiment_params["xi"]/GDP_per_capita,
						"vg_life":dynModel.econ_params["econ_cost_death"][ag],
						"chi+vg_life":(dynModel.experiment_params["xi"]+dynModel.econ_params["econ_cost_death"][ag])/GDP_per_capita,
						"p_H": dynModel.parameters["seir-groups"][ag]['parameters']['p_H'],
						"p_ICU": dynModel.parameters["seir-groups"][ag]['parameters']['p_ICU'],
						"p_ss": dynModel.parameters["seir-groups"][ag]['parameters']['p_ss'],
						"p_D": p_death,
						"econ_gradient_normalized":dynModel.econ_gradients_normalized[act][ag],
						"econ_gradient_not_normalized":dynModel.econ_gradients[act][ag],
						"I_perc":np.sum([state[other_group]["I"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
						"R_perc":np.sum([state[other_group]["R"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
						"H_perc":np.sum([state[other_group]["H"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
						"ICU_utilization":np.sum([state[other_group]["ICU"] for other_group in age_groups])/dynModel.experiment_params["icus"],
						"contacts_received":np.sum([dynModel.parameters["seir-groups"][ag]["contacts_spc"][act][other_group] for other_group in age_groups]),
						"contacts_received_into_deaths":np.sum([dynModel.parameters["seir-groups"][ag]["contacts_spc"][act][other_group]* p_death for other_group in age_groups]),
						"infections_received":np.sum([dynModel.parameters["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"]) for other_group in age_groups]),
						"deaths_received":np.sum([dynModel.parameters["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"])*p_death for other_group in age_groups]),
						"deaths_received_econ":np.sum([dynModel.parameters["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"])*p_death*(dynModel.experiment_params["xi"]+dynModel.econ_params["econ_cost_death"][ag]) for other_group in age_groups]),
						"contacts_given":np.sum([dynModel.parameters["seir-groups"][other_group]["contacts_spc"][act][ag] for other_group in age_groups]),
						"contacts_given_into_deaths":np.sum([dynModel.parameters["seir-groups"][other_group]["contacts_spc"][act][ag] * p_death for other_group in age_groups]),
						"infections_given":np.sum([dynModel.parameters["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"]) for other_group in age_groups]),
						"deaths_given":np.sum([dynModel.parameters["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"]) * p_death for other_group in age_groups]),
						"deaths_given_econ":np.sum([dynModel.parameters["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"]) * p_death *(dynModel.experiment_params["xi"]+dynModel.econ_params["econ_cost_death"][other_group]) for other_group in age_groups]),
						"econ_contacts_ratio": dynModel.econ_gradients_normalized[act][ag] / (np.sum([dynModel.parameters["seir-groups"][other_group]["contacts_spc"][act][ag] for other_group in age_groups])),
						"econ_contacts_death_cost_ratio":dynModel.econ_gradients_normalized[act][ag] / (dynModel.parameters["seir-groups"][ag]['parameters']['beta'][0] * np.sum([dynModel.parameters["seir-groups"][other_group]["contacts_spc"][act][ag] for other_group in age_groups]) * (dynModel.experiment_params["xi"]+dynModel.econ_params["econ_cost_death"][ag]) * dynModel.parameters["seir-groups"][ag]['parameters']['sigma'] *dynModel.parameters["seir-groups"][ag]['parameters']['mu']),
						"ICU_new_daily_admissions": newICU,
						"ICU_seven_day_avg_daily_admissions": sevendays_avg_newICU,
						"Infections_new_rate_over_seven_days": incidence_rate,	
						"Infections_new_rate_over_seven_days_elderly": incidence_rate_old,
						"S": state[ag]["S"],
						"E": state[ag]["E"],
						"I": state[ag]["I"],
						"Ia": state[ag]["Ia"],
						"Ims": state[ag]["Ims"],
						"Ips": state[ag]["Ips"],
						"Iss": state[ag]["Iss"],
						"ICU": state[ag]["ICU"],
						"H": state[ag]["H"],
						"R": state[ag]["R"], 
						"Rq": state[ag]["Rq"],
						"D": state[ag]["D"]
					})
    
    return features

def get_alphas(dynModel, state, incidence_rate, incidence_rate_old, sevendays_avg_newICU, newICU, t, tree_file):
    feature_set_name = tree_file.split('_variables_')[1][:-7]

    feature_set = get_feature_set_from_name(feature_set_name)
   

    rel_activities = ['leisure','other','school','transport','work']
    all_activities = ['home','leisure','other','school','transport','work']
    
    trees = {}
    for act in rel_activities:
        with open(f"decision_tree/trees_pickles/{act}_{tree_file}", "rb") as pickle_file:
            trees[act] = pickle.load(pickle_file)
    
    features = get_features(dynModel, state, incidence_rate, incidence_rate_old,sevendays_avg_newICU, newICU, t, tree_file)

    alphas = {}
    for act in all_activities:
        for ag in dynModel.age_groups:
            alphas[ag] = {}
            alphas[ag][act] = 1.0

    for act in rel_activities:
        for f in features[act]:
            ag = number_to_age_group(f['age'])
            
            alphas[ag][act] = trees[act].predict(pd.DataFrame([f])[ feature_set])

    return alphas



# Runs the policy defined by the trees on dynModel
def run_tree_policy(dynModel, tree_file):

    T = dynModel.time_steps

    sevendays_newICU = []
    sevendays_newInf = []
    sevendays_newInf_old = []

    for t in range(90):
        state = dynModel.get_state(t)

        if (t==0):
            initial_population = np.sum([state[ag]["N"] for ag in dynModel.age_groups])
            initial_population_old = np.sum([state[ag]["N"] for ag in ['age_group_60_69', 'age_group_70_79', 'age_group_80_plus']])


        ## Compute new infections 
        # Find incidence rate over last 7 days in overall population
        newInfections = np.sum([dynModel.groups[ag].parameters['sigma']*state[ag]["E"] for ag in dynModel.age_groups])
        sevendays_newInf.append(newInfections)
        # Drop first entry if larger than 7
        if len(sevendays_newInf)>7:
            sevendays_newInf = sevendays_newInf[1:8]
        
        incidence_rate = np.sum(sevendays_newInf)/initial_population


        # Find incidence rate over last 7 days in population aged 60+
        newInfections_old = np.sum([dynModel.groups[ag].parameters['sigma']*state[ag]["E"] for ag in ['age_group_60_69', 'age_group_70_79', 'age_group_80_plus']])
        sevendays_newInf_old.append(newInfections_old)
        # Drop first entry if larger than 7
        if len(sevendays_newInf_old)>7:
            sevendays_newInf_old = sevendays_newInf_old[1:8]

        incidence_rate_old = np.sum(sevendays_newInf_old)/initial_population_old


        ## Compute new ICU admissions
        newICU = np.sum([dynModel.groups[ag].parameters['mu']*(dynModel.groups[ag].parameters['p_ICU']*state[ag]["I"] + ((dynModel.groups[ag].parameters['p_ICU'])/(dynModel.groups[ag].parameters['p_H']+dynModel.groups[ag].parameters['p_ICU']))*state[ag]["Iss"]) for ag in dynModel.age_groups])
        sevendays_newICU.append(newICU)
        # Drop first entry if larger than 7
        if len(sevendays_newICU)>7:
            sevendays_newICU = sevendays_newICU[1:8]
		
        sevendays_avg_newICU = np.sum(sevendays_newICU)/len(sevendays_newICU)


        alphas = get_alphas(dynModel, state, incidence_rate, incidence_rate_old, sevendays_avg_newICU, newICU, t, tree_file)

        m_tests = {ag:0 for ag in dynModel.age_groups}
        a_tests = {ag:0 for ag in dynModel.age_groups}

        dynModel.take_time_step(m_tests, a_tests, alphas)
    
    dynModel.take_end_steps()
    
    return 0


def main():
    pass

    # Global variables
    simulation_params = {
    'dt':1.0,
    'days': 90.0,
    'region': "fitted",
    'heuristic': 'benchmark',
    'mixing_method': {'name': 'multi'}
    }
    
    simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

    init_param = ["multiplier_beta", "multiplier_p_icu", "multiplier_p_d","multiplier_lambda_h", "multiplier_lambda_icu", "alpha_other","l_school_march", "l_school_may", "l_school_july", "l_school_september", "alpha_mixing", "econ_value", "l_work_april"]

    age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49', 'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']


    cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
        'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]

    activities = ['home','leisure','other','school','transport','work']
    
    rel_activities = ['leisure','other','school','transport','work']

    # Define time variables
    simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))


    # Read the three intialization files
    # 
    
    group_params_file = "../parameters/fitted.yaml"
    initialization_file = "../initialization/oct21.yaml"
    econ_param_file = "../parameters/econ.yaml"

	# Read group parameters
    with open(group_params_file) as file:
        universe_params = yaml.load(file, Loader=yaml.FullLoader)

    # Read initialization
    with open(initialization_file) as file:
        initialization = yaml.load(file, Loader=yaml.FullLoader)
        start_day = 0

    # Read econ parameters
    with open(econ_param_file) as file:
        econ_params = yaml.load(file, Loader=yaml.FullLoader)








if __name__ == "__main__":
    main()
