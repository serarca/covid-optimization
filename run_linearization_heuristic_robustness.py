# -*- coding: utf-8 -*-
import yaml
from inspect import getsourcefile
import os.path
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
sys.path.insert(0, current_dir+"/heuristics")

import random

import numpy as np
from group import SEIR_group, DynamicalModel
import linearization

import math
import pprint
from time import time

try:
    import cPickle as pickle
except:
    import pickle
import pandas as pd
import logging
import itertools as it

#from joblib import Parallel, delayed

init_param = ["multiplier_beta", "multiplier_p_icu", "multiplier_p_d","multiplier_lambda_h", "multiplier_lambda_icu", "alpha_other","l_school_march", "l_school_may", "l_school_july", "l_school_september", "alpha_mixing", "econ_value", "l_work_april"]

def main():

    instance_index = 0

    if len(sys.argv) > 1:
        instance_index = int(sys.argv[1])

    

    # 30 * 37199.03
    # Some paramters to test the linearization heuristic
    scaling = 10000
    money_scaling = 1000
    xi_mult_values = [0,10,25,50,100,150]
    # chain(range(0,200,5), range(0, 1000, 10))
    testing_values = [0]
    # [0, 30000, 60000, 120000]
    icu_values = [2900]
    # [2000, 2300, 2600, 2900, 3200]
    random_instances=range(0,40)


    init_param_to_vary = init_param[8:9]


    # eta_inteval_lb = 0
    # eta_inteval_ub = 0.2

    # delta_school_interval_lb = 0
    # delta_school_interval_ub = 1

    # lock_work_april_interval_lb = 48.51
    # lock_work_april_interval_ub = 68.51

    # econ_act_april_interval_lb = 0.19182
    # econ_act_april_interval_ub = 0.23444666666   
    

    params_to_try = {
        "delta_schooling":[0.5],
        "xi":[mult * 37199.03 * scaling / money_scaling for mult in xi_mult_values],
        "icus":[ic / scaling for ic in icu_values],
        "mtests":[test_cap / scaling for test_cap in testing_values],
        # "atests":[test_cap / scaling for test_cap in testing_values],
        "frequencies":[(7,14)],
        "region":["fitted-scaled"], 
        "econ": ["econ-scaled"],
        "init": ["oct21-scaled"],
        "eta":[0.1],
        "trust_region_radius":[0.05],
        "max_inner_iterations_mult":[2],
        "initial_uhat":["activity_gradient", "age_group_gradient", "dynamic_gradient"],
        "random_instances":random_instances,
        "init_param_to_vary": init_param_to_vary
        # "full_lockdown", "full_open","dynamic_gradient",
    }


    all_instances = list(it.product(*(params_to_try[param] for param in params_to_try)))


    # params_to_try = {
    #     "delta_schooling":[0.5],
    #     "xi":[30 * 37199.03],
    #     "icus":[3000],
    #     "tests":[0],
    #     "frequencies":[(1,1)],
    #     "region":["one_group_fitted"], 
    #     "econ": ["one_group_econ"],
    #     "init": ["60days_one_group"],
    #     "eta":[0.1]
    # }

    n_days = 90
    groups = "all"
    start_day = 0
    optimize_bouncing = False

    print(len(all_instances))


    

    # Final time step is used if we want to evaluate 
    # the hueristic at any time before the n_days
    final_time_step = n_days
    
    # For names of regions see the "parameters" folder
    # region = 'fitted'

    delta = all_instances[instance_index][0]
    xi = all_instances[instance_index][1]
    icus = all_instances[instance_index][2]
    mtests = all_instances[instance_index][3]
    atests = 0

    # atests = all_instances[instance_index][4]
    print(all_instances[instance_index])
    test_freq = all_instances[instance_index][4][0]
    lockdown_freq = all_instances[instance_index][4][1]
    region = all_instances[instance_index][5]
    econ = all_instances[instance_index][6]
    init = all_instances[instance_index][7]
    eta = all_instances[instance_index][8]
    trust_region_radius = all_instances[instance_index][9]
    max_inner_iterations_mult = all_instances[instance_index][10]
    initial_uhat = all_instances[instance_index][11]
    random_instance = all_instances[instance_index][12]
    init_param_varying = all_instances[instance_index][13]
    
    scaling_econ_param(scaling, money_scaling, groups, random_instance, init_param_varying)
    scaling_fitted(scaling, money_scaling, groups, random_instance, init_param_varying)
    scaling_init(scaling, groups, random_instance, init_param_varying)


    random.seed(random_instance)
    # eta = random.uniform(eta_inteval_lb, eta_inteval_ub)
    # delta = random.uniform(delta_school_interval_lb, delta_school_interval_ub)

    # econ_activ_april = random.uniform(econ_act_april_interval_lb, econ_act_april_interval_ub)
    # lock_work_april = random.uniform(lock_work_april_interval_lb, lock_work_april_interval_ub)

    # nu = econ_activ_april / (1 + lock_work_april)
    # gamma = 1 - nu

    targetActivities = True
    targetGroups = True
    targetTests = True

    if initial_uhat == "time_gradient":
        targetActivities = False
        targetGroups = False
    
    if initial_uhat == "age_group_gradient":
        targetActivities = False
    
    if initial_uhat == "activity_gradient":
        targetGroups = False


    run_lin_heur_and_save_yaml(delta, xi, icus, mtests, atests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, scaling, money_scaling, random_instance, init_param_varying, targetActivities, targetGroups, targetTests)
    
    
    # run_lin_heur_and_pickle_dynModel(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day)

    # run_pickled_dynModels_prop_bouncing(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day)

    # for heur in ["", "_Prop_Bouncing"]:
    #     load_pickle_and_create_yaml(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, scaling, money_scaling, heur)


def run_lin_heur_and_save_yaml(delta, xi, icus, mtests, atests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, scaling, money_scaling, random_instance, init_param_varying, targetActivities, targetGroups, targetTests):
    ''' Runs the linearization heuristic with the experiment parameters passed as arguments and saves the resulting dynamical model as a pickle object.'''

    experiment_params = {
        'delta_schooling':delta,
        'xi':xi,
        'icus':icus,
        'test_freq': test_freq,
        'lockdown_freq': lockdown_freq
    }

    simulation_params = {
        'dt':1.0,
        'region': region,
        'quar_freq': 1,
        'num_days' : n_days,
        'initial_infected_count' : 1,
        'mixing_method' : {
            "name":"mult"},
        'mtest_cap' : mtests,
        'atest_cap' : atests,
        'heuristic': 'linearization',
        'transport_lb_work_fraction': 0.25,
        'econ': econ,
        'init': init,
        'eta': eta
    }

    dynModel = run_linearization_heuristic(simulation_params, experiment_params, start_day, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, random_instance, init_param_varying, targetActivities, targetGroups, targetTests)

    result = {
            "lockdown_heuristic":f"linearization_heuristic_optBouncing={optimize_bouncing}_initial_uhat={initial_uhat}_initParamVarying={init_param_varying}",
            "groups":groups,
            "experiment_params":{
                "delta_schooling":delta,
                "xi":(xi/scaling) * money_scaling, 
                "icus":icus * scaling,
                "n_a_tests":atests * scaling,
                "n_m_tests":mtests * scaling,
                "start_day":start_day,
                "T":n_days,
                "eta":eta,
                "lockdown_freq":lockdown_freq,
                "test_freq":test_freq
            },
            "testing_heuristic":f"linearization_heuristic",
            "results":{
                "economics_value":float(dynModel.get_total_economic_value()) * money_scaling,
                "deaths":float(dynModel.get_total_deaths()) * scaling,
                "reward":float(dynModel.get_total_reward()),
            },
            "policy":dynModel.lockdown_controls,
            "a_tests":[{g: test * scaling for g,test in a.items()} for a in dynModel.a_tests_controls],
            "m_tests":[{g: test * scaling for g,test in m.items()}  for m in dynModel.m_tests_controls]
    }

    result["filename"] = f"{result['lockdown_heuristic']}/xi-{result['experiment_params']['xi']}_icus-{result['experiment_params']['icus']}_testing-{result['testing_heuristic']}_natests-{result['experiment_params']['n_a_tests']}_nmtests-{result['experiment_params']['n_m_tests']}_T-{result['experiment_params']['T']}_startday-{result['experiment_params']['start_day']}_groups-{result['groups']}_dschool-{result['experiment_params']['delta_schooling']}_eta-{result['experiment_params']['eta']}_lockdownFreq-{result['experiment_params']['lockdown_freq']}_testingFreq-{result['experiment_params']['test_freq']}_initParamVarying-{init_param_varying}_randomInstance-{random_instance}"

    fn =  f"benchmarks/results/{result['filename']}.yaml"

    if not os.path.exists(f"benchmarks/results/{result['lockdown_heuristic']}"):
        os.makedirs(f"benchmarks/results/{result['lockdown_heuristic']}")
    
    with open(fn, 'w') as file:
        yaml.dump(result, file)



def run_linearization_heuristic(simulation_params, experiment_params, start_day, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, random_instance, init_param_varying, targetActivities, targetGroups, targetTests):
    ''' Takes a set of simulation_params and experiment parameters (delta_school, emotional cost of deaths (xi), max icus, max tests, testing and lockdown frequencies) and a set of simulation paramters (required by the constructor in group.py), creates a dynamical system, runs the linearization heuristic and returns the dynamical system after running the heuristic. 
    '''

    start_time = time()

    # Define time variables
    num_time_periods = int(math.ceil(simulation_params["num_days"]/simulation_params["dt"]))


    # Read group parameters
    with open("parameters/simulations/"+simulation_params["region"]+f"_{init_param.index(init_param_varying)}_{random_instance}.yaml","r") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        while file is None:
            print("Failed reading fitted param")
            file = open("parameters/simulations/"+simulation_params["region"]+f"_{init_param.index(init_param_varying)}_{random_instance}.yaml","r")
        
        universe_params = yaml.load(file, Loader=yaml.FullLoader)
        
        while universe_params is None:
            print("Failed converting to yaml fitted param")
            universe_params = yaml.load(file, Loader=yaml.FullLoader)

        # Read initialization
    with open(f"parameters/simulations/{simulation_params['init']}_{init_param.index(init_param_varying)}_{random_instance}.yaml","r") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        while file is None:
            print("Failed reading init param")
            file = open(f"parameters/simulations/{simulation_params['init']}_{init_param.index(init_param_varying)}_{random_instance}.yaml","r")

        initialization = yaml.load(file, Loader=yaml.FullLoader)
        while initialization is None:
            print("Failed converting to yaml init param")
            initialization = yaml.load(file, Loader=yaml.FullLoader)
    
    # Read econ parameters
    with open(f"parameters/simulations/{simulation_params['econ']}_{init_param.index(init_param_varying)}_{random_instance}.yaml","r") as file:
        while file is None:
            print("Failed reading econ param")
            file = open(f"parameters/simulations/{simulation_params['econ']}_{init_param.index(init_param_varying)}_{random_instance}.yaml","r")

        econ_params = yaml.load(file, Loader=yaml.FullLoader)
        while econ_params is None:
            print("Failed converting to yaml econ param")
            econ_params = yaml.load(file, Loader=yaml.FullLoader)
    
    # print(universe_params)

    # Define mixing method
    mixing_method = universe_params['mixing']

    eta = simulation_params["eta"]
    

    dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], num_time_periods, mixing_method, start_day, eta)

    # add parameters for testing capacity
    dynModel.parameters['global-parameters']['C_mtest'] = simulation_params['mtest_cap']
    dynModel.parameters['global-parameters']['C_atest'] = simulation_params['atest_cap']


    # Change eta, nu, and gamma
    # dynModel.econ_params["employment_params"]["eta"] = simulation_params["eta"]
    # dynModel.econ_params["employment_params"]["nu"] = 1 - gamma - simulation_params["eta"]
    # dynModel.econ_params["employment_params"]["gamma"] = gamma

    # targetActivities=True
    # targetGroups=True
    # targetTests=True
    deltaFairnessOne=False
    deltaFair=0.1
    optimizeLockdowns=True
    averageLockConst=False
    pLock=1
    optimizeOnlyDeaths=False

    linearization.run_heuristic_linearization(dynModel, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, targetActivities, targetGroups, targetTests, deltaFairnessOne, deltaFair, optimizeLockdowns, averageLockConst, pLock, optimizeOnlyDeaths, random_instance, init_param_varying)


    end_time = time()

    print("Total running time for {} days is {}".format(simulation_params['num_days'], end_time - start_time))

    dynModel.print_stats()

    return dynModel


def scaling_econ_param(scaling, money_scaling, groups, random_instance, init_param_varying):
    # Import data
    if groups == "all":
        old_econ = yaml.load(open( f"parameters/simulations/econ_{init_param.index(init_param_varying)}_{random_instance}.yaml", "rb" ),Loader=yaml.FullLoader)

    
    # scaling = 1000.0
    # money_scaling = 10000.0

    scaled_econ = dict(old_econ)

    # Scale Econ cost of death
    for group in scaled_econ["econ_cost_death"]:
        scaled_econ["econ_cost_death"][group] = (scaled_econ["econ_cost_death"][group] * scaling / money_scaling)

    # Scale employment param

    for group in scaled_econ["employment_params"]["v"]:
        scaled_econ["employment_params"]["v"][group]["leisure"] = scaled_econ["employment_params"]["v"][group]["leisure"] * scaling / money_scaling
        scaled_econ["employment_params"]["v"][group]["other"] = scaled_econ["employment_params"]["v"][group]["other"] * scaling / money_scaling
        scaled_econ["employment_params"]["v"][group]["transport"] = scaled_econ["employment_params"]["v"][group]["transport"] * scaling / money_scaling

    # Scale schooling params

    for group in scaled_econ["schooling_params"]:
        scaled_econ["schooling_params"][group] = scaled_econ["schooling_params"][group] * scaling / money_scaling


    if groups == "all":
        with open(f'parameters/simulations/econ-scaled_{init_param.index(init_param_varying)}_{random_instance}.yaml', 'w') as file:
            yaml.dump(scaled_econ, file)
    
    

def scaling_init(scaling, groups, random_instance, init_param_varying):
    # Import data
    if groups == "all":
        old_init = yaml.load(open( f"parameters/simulations/oct21_{init_param.index(init_param_varying)}_{random_instance}.yaml", "rb" ), Loader=yaml.FullLoader)


    
    # scaling = 1000.0

    # Construct initialization
    scaled_init_dict = {}
    for group in old_init:
        scaled_init_dict[group] = {
                "S": old_init[group]["S"] / scaling,
                "E": old_init[group]["E"] / scaling,
                "I": old_init[group]["I"] / scaling,
                "R": old_init[group]["R"] / scaling,
                "Ia": old_init[group]["Ia"] / scaling,
                "Ips": old_init[group]["Ips"] / scaling,
                "Ims": old_init[group]["Ims"] / scaling,
                "Iss": old_init[group]["Iss"] / scaling,
                "Rq": old_init[group]["Rq"] / scaling,
                "H": old_init[group]["H"] / scaling,
                "ICU": old_init[group]["ICU"] / scaling,
                "D": old_init[group]["D"] / scaling,
        }

    if groups == "all":
        with open(f'parameters/simulations/oct21-scaled_{init_param.index(init_param_varying)}_{random_instance}.yaml', 'w') as file:
            yaml.dump(scaled_init_dict, file)


    


def scaling_fitted(scaling, money_scaling, groups, random_instance, init_param_varying):

    if groups == "all":
        # Import data
        old_fitted = yaml.load(open( f"parameters/simulations/fitted_{init_param.index(init_param_varying)}_{random_instance}.yaml", "rb" ), Loader=yaml.FullLoader)
    # scaling = 1000.0

    scaled_fitted = dict(old_fitted)

    # Scale global_param
    scaled_fitted["global-parameters"]["C_H"] = scaled_fitted["global-parameters"]["C_H"] / scaling

    scaled_fitted["global-parameters"]["C_ICU"] = scaled_fitted["global-parameters"]["C_ICU"] / scaling



    if groups == "all":
        with open(f'parameters/simulations/fitted-scaled_{init_param.index(init_param_varying)}_{random_instance}.yaml', 'w') as file:
            yaml.dump(scaled_fitted, file)






if __name__ == "__main__":
    main()
