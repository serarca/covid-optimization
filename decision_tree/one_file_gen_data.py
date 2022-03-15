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
os.system('module load math')
os.system('module load python/3.6.1')
os.system('module load gurobipy3')
os.system('module load viz')
os.system('module load py-matplotlib/3.2.1_py36')
os.system('module load py-pandas/0.23.0_py36')
os.system('module load py-scipy/1.4.1_py36')
os.system('module load py-numpy/1.19.2_py36')
os.system('ml load py-schwimmbad')



from group import SEIR_group, DynamicalModel
import math
import pprint
import pandas as pd
import pickle
import numpy as np
from fast_group import FastDynamicalModel
from aux import *
from scipy.optimize import Bounds,minimize,LinearConstraint
from os import listdir
from os.path import isfile, join
from schwimmbad import JoblibPool



def extract_features(f, init_param, age_groups, cont, activities, rel_activities, group_params_file, initialization_file, econ_param_file, instances_folder, simulation_params):
    # if instances_folder == "../benchmarks/results/linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient/":
    # 	if i<=906:
    # 		print(f"File number: {i} not processed for dynamice_gradient")
    # 		continue

    ## If instance is multiple_init_cond change param folders
    if "multiple_init_cond" in instances_folder:
        randomInstance = f.split("-")[-1]
        group_params_file = f"../parameters/multiple_initial_conditions/fitted_{randomInstance}"
        initialization_file = f"../parameters/multiple_initial_conditions/oct21_{randomInstance}"
        econ_param_file = f"../parameters/multiple_initial_conditions/econ_{randomInstance}"


    ## If instance is Robust
    if "initParamVarying" in instances_folder:
        randomInstance = f.split("-")[-1]
        initParamVarying = instances_folder.split("=")[-1][:-1]
        group_params_file = f"../parameters/simulations/fitted_{init_param.index(initParamVarying)}_{randomInstance}"
        initialization_file = f"../parameters/simulations/oct21_{init_param.index(initParamVarying)}_{randomInstance}"
        econ_param_file = f"../parameters/simulations/econ_{init_param.index(initParamVarying)}_{randomInstance}"
    
    



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

    
    
    features = []

#		if i >= max_number_files:
#			break
    
    # print(i/len(files_to_process))
    with open(instances_folder+f) as file:
        file_result = yaml.load(file, Loader=yaml.UnsafeLoader)

    experiment_params = file_result["experiment_params"]
    mixing_method = universe_params["mixing"]

    dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day, experiment_params["eta"], extra_data = True)

    sevendays_newICU = []
    sevendays_newInf = []
    sevendays_newInf_old = []


    for t in range(simulation_params['time_periods']):

        result = dynModel.take_time_step(file_result['m_tests'][t], file_result['a_tests'][t], file_result['policy'][t])
        
        if t%14 == 0:

            print(f"t is {t}")
            
            state = dynModel.get_state(t)
            

            if (t==0):
                initial_population = np.sum([state[ag]["N"] for ag in age_groups])
                initial_population_old = np.sum([state[ag]["N"] for ag in ['age_group_60_69', 'age_group_70_79', 'age_group_80_plus']])


            ## Compute new infections 
            # Find incidence rate over last 7 days in overall population
            newInfections = np.sum([dynModel.groups[ag].parameters['sigma']*state[ag]["E"] for ag in age_groups])
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
            ######
            ## Compute new ICU admissions
            newICU = np.sum([dynModel.groups[ag].parameters['mu']*(dynModel.groups[ag].parameters['p_ICU']*state[ag]["I"] + ((dynModel.groups[ag].parameters['p_ICU'])/(dynModel.groups[ag].parameters['p_H']+dynModel.groups[ag].parameters['p_ICU']))*state[ag]["Iss"]) for ag in age_groups])
            sevendays_newICU.append(newICU)
            # Drop first entry if larger than 7
            if len(sevendays_newICU)>7:
                sevendays_newICU = sevendays_newICU[1:8]
            sevendays_avg_newICU = np.sum(sevendays_newICU)/len(sevendays_newICU)

            econ_gradients = dynModel.econ_gradients

            # Extract the features
            for act in rel_activities:
                for ag in age_groups:
                    p_death = 1 / (universe_params["seir-groups"][ag]['parameters']['lambda_H_R'] / universe_params["seir-groups"][ag]['parameters']['lambda_H_D'] + 1)

                    
                    features.append({
                        "age_group":ag,
                        "activity":act,
                        "t":t,
                        "lockdown":file_result['policy'][t][ag][act],
                        "icu_capacity":experiment_params["icus"],
                        "theta_schooling": experiment_params["delta_schooling"],
                        "beta": universe_params["seir-groups"][ag]['parameters']['beta'][0],
                        "alpha_1_2": universe_params["mixing"]['param_alpha'],
                        "nu_other_act":dynModel.econ_params["employment_params"]["eta"],
                        "nu_work":dynModel.econ_params["employment_params"]['nu'],
                        "chi":experiment_params["xi"],
                        "vg_life":econ_params["econ_cost_death"][ag],
                        "xi+vg_life":experiment_params["xi"]+econ_params["econ_cost_death"][ag],
                        "p_H": universe_params["seir-groups"][ag]['parameters']['p_H'],
                        "p_ICU": universe_params["seir-groups"][ag]['parameters']['p_ICU'],
                        "p_ss": universe_params["seir-groups"][ag]['parameters']['p_ss'],
                        "p_D": p_death,
                        "econ_gradient_normalized":dynModel.econ_gradients_normalized[act][ag],
                        "econ_gradient_not_normalized":econ_gradients[act][ag],
                        "I_perc":np.sum([state[other_group]["I"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
                        "R_perc":np.sum([state[other_group]["R"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
                        "H_perc":np.sum([state[other_group]["H"] for other_group in age_groups])/np.sum([state[other_group]["N"]+state[other_group]["Rq"] for other_group in age_groups]),
                        "ICU_utilization":np.sum([state[other_group]["ICU"] for other_group in age_groups])/experiment_params["icus"],
                        "contacts_received":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group] for other_group in age_groups]),
                        "contacts_received_into_deaths":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group]* p_death for other_group in age_groups]),
                        "infections_received":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"]) for other_group in age_groups]),
                        "deaths_received":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"])*p_death for other_group in age_groups]),
                        "deaths_received_econ":np.sum([universe_params["seir-groups"][ag]["contacts_spc"][act][other_group]*state[other_group]["I"]/(state[other_group]["Rq"]+state[other_group]["N"])*p_death*(experiment_params["xi"]+econ_params["econ_cost_death"][ag]) for other_group in age_groups]),
                        "contacts_given":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag] for other_group in age_groups]),
                        "contacts_given_into_deaths":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag] * p_death for other_group in age_groups]),
                        "infections_given":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"]) for other_group in age_groups]),
                        "deaths_given":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"]) * p_death for other_group in age_groups]),
                        "deaths_given_econ":np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag]*state[ag]["I"]/(state[ag]["Rq"]+state[ag]["N"]) * p_death *(experiment_params["xi"]+econ_params["econ_cost_death"][other_group]) for other_group in age_groups]),
                        "econ_contacts_ratio": dynModel.econ_gradients_normalized[act][ag] / (np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag] for other_group in age_groups])),
                        "econ_contacts_death_cost_ratio":dynModel.econ_gradients_normalized[act][ag] / (universe_params["seir-groups"][ag]['parameters']['beta'][0] * np.sum([universe_params["seir-groups"][other_group]["contacts_spc"][act][ag] for other_group in age_groups]) * (experiment_params["xi"]+econ_params["econ_cost_death"][ag]) * universe_params["seir-groups"][ag]['parameters']['sigma'] *universe_params["seir-groups"][ag]['parameters']['mu']),
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
                        "D": state[ag]["D"],
                        "instance": instances_folder+f
                    })
    return features
                


def append_instances_to_csv(csv_file, instances_folder):


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

    #TODO: Read CSV file if it exists, extract all instances that have been already read, filter files below for those not processed

    processed_files = []

    if os.path.exists(csv_file+".csv"):
        old_dataset = pd.read_csv(csv_file+".csv")
        processed_files = set(old_dataset["instance"].unique())

    # Read files
    files_to_process = [f for f in listdir(instances_folder) if (isfile(join(instances_folder, f)) and f[0:2]=="xi") and (instances_folder+f not in processed_files)]

     

    print(len(files_to_process))
    

    # Default parameter files (for dynamic-gradient)

    group_params_file = "../parameters/fitted.yaml"
    initialization_file = "../initialization/oct21.yaml"
    econ_param_file = "../parameters/econ.yaml"

    init_file = 0
    batch_size = 500

    while init_file < len(files_to_process):
        last_file = min(init_file + batch_size, len(files_to_process))

        batch_of_files = files_to_process[init_file : last_file]

        with JoblibPool(20) as pool:
            list_of_list_of_features = list(pool.map(lambda x: extract_features(x, init_param, age_groups, cont, activities, rel_activities, group_params_file, initialization_file, econ_param_file, instances_folder, simulation_params), batch_of_files))
        


        list_of_features = [feature for inst in list_of_list_of_features for feature in inst]

        if not os.path.exists(csv_file+".csv"):
            useHeader = True
        else:
            useHeader = False
                
        pd.DataFrame(list_of_features).to_csv(csv_file+".csv", mode='a', header=useHeader, index=False)

        with open("log.txt", 'w') as file:
            file.write(f"Just wrote files {init_file} to {last_file} from folder {instances_folder} witih {len(files_to_process)} total files.")
        
        init_file = last_file



def main():
    
    csv_file = "dataset"


    ## Dynamic-Gradient-Instances

    # instances_folder = "../benchmarks/results/linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient_multiple_init_cond-HighBeta/"

    # append_instances_to_csv(csv_file, instances_folder)

    instances_folder = "../benchmarks/results/linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient_multiple_init_cond/"

    append_instances_to_csv(csv_file, instances_folder)

    # instances_folder = "../benchmarks/results/linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient/"

    # append_instances_to_csv(csv_file, instances_folder)

    init_param = ["multiplier_beta", "multiplier_p_icu", "multiplier_p_d","multiplier_lambda_h", "multiplier_lambda_icu", "alpha_other","l_school_march", "l_school_may", "l_school_july", "l_school_september", "alpha_mixing", "econ_value", "l_work_april"]

    # for init_param_varying in init_param:
    #     instances_folder = f"../benchmarks/results/linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient_initParamVarying={init_param_varying}/"

    #     append_instances_to_csv(csv_file, instances_folder)


    


if __name__ == "__main__":
    main()

