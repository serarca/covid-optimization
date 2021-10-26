# -*- coding: utf-8 -*-
import yaml
from inspect import getsourcefile
import os.path
import sys
import argparse
import matplotlib
from time import time
import math
matplotlib.use('Agg')
import matplotlib.pyplot as plt
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
sys.path.insert(0, current_dir+"/heuristics")
sys.path.insert(0, current_dir+"/decision_tree")


import tree_policy_benchmark


import random

import numpy as np
from group import SEIR_group, DynamicalModel

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

def main():

    instance_index = 0

    if len(sys.argv) > 1:
        instance_index = int(sys.argv[1])

    # 30 * 37199.03
    # Some paramters to test the linearization heuristic
    
    xi_mult_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50,75, 100, 125, 150, 175, 200, 250, 300, 400, 500, 600, 700, 800, 900] 
    # range(0,205,5)
    testing_values = [0]
    # [0, 30000, 60000, 120000]
    icu_values = [2900]
    # [2000, 2300, 2600, 2900, 3200]

    params_to_try = {
        "delta_schooling":[0.5],
        "xi":[mult * 37199.03  for mult in xi_mult_values],
        "icus":[ic for ic in icu_values],
        "mtests":[test_cap  for test_cap in testing_values],
        # "atests":[test_cap / scaling for test_cap in testing_values],
        "frequencies":[(7,14)],
        "region":["fitted-scaled"], 
        "econ": ["econ-scaled"],
        "init": ["oct21-scaled"],
        "eta":[0.1],
        "feature_set": ["ENGINEERED_1_MINUS", "ENGINEERED_1_PLUS", "ORIGINAL", "ORIGINAL_PLUS", "SIMPLE", "SIMPLE_PLUS"], 
        "tree_depth": [5]
        # "trust_region_radius":[0.05],
        # "max_inner_iterations_mult":[2],
        # "initial_uhat":["dynamic_gradient"]
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


    # scaling_econ_param(scaling, money_scaling, groups)
    # scaling_fitted(scaling, money_scaling, groups)
    # scaling_init(scaling, groups)

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
    feature_set = all_instances[instance_index][9]
    tree_depth = all_instances[instance_index][10]
    # trust_region_radius = all_instances[instance_index][9]
    # max_inner_iterations_mult = all_instances[instance_index][10]
    # initial_uhat = all_instances[instance_index][11]
    

    run_tree_heur_and_save_yaml(delta, xi, icus, mtests, atests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, feature_set, tree_depth)
    
    
    # run_lin_heur_and_pickle_dynModel(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day)

    # run_pickled_dynModels_prop_bouncing(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day)

    # for heur in ["", "_Prop_Bouncing"]:
    #     load_pickle_and_create_yaml(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, scaling, money_scaling, heur)


def run_tree_heur_and_save_yaml(delta, xi, icus, mtests, atests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, feature_set, tree_depth):
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

    dynModel = run_tree_heuristic(simulation_params, experiment_params, start_day, feature_set, tree_depth)

    result = {
            "lockdown_heuristic":f"tree_heuristic_featureSet={feature_set}_treeDepth={tree_depth}",
            "groups":groups,
            "experiment_params":{
                "delta_schooling":delta,
                "xi":xi, 
                "icus":icus ,
                "n_a_tests":atests,
                "n_m_tests":mtests,
                "start_day":start_day,
                "T":n_days,
                "eta":eta,
                "lockdown_freq":lockdown_freq,
                "test_freq":test_freq
            },
            "testing_heuristic":f"None",
            "results":{
                "economics_value":float(dynModel.get_total_economic_value()),
                "deaths":float(dynModel.get_total_deaths()),
                "reward":float(dynModel.get_total_reward()),
            },
            "policy":dynModel.lockdown_controls,
            "a_tests":[{g: test  for g,test in a.items()} for a in dynModel.a_tests_controls],
            "m_tests":[{g: test  for g,test in m.items()}  for m in dynModel.m_tests_controls]
    }

    result["filename"] = f"{result['lockdown_heuristic']}/xi-{result['experiment_params']['xi']}_icus-{result['experiment_params']['icus']}_testing-{result['testing_heuristic']}_natests-{result['experiment_params']['n_a_tests']}_nmtests-{result['experiment_params']['n_m_tests']}_T-{result['experiment_params']['T']}_startday-{result['experiment_params']['start_day']}_groups-{result['groups']}_dschool-{result['experiment_params']['delta_schooling']}_eta-{result['experiment_params']['eta']}_lockdownFreq-{result['experiment_params']['lockdown_freq']}_testingFreq-{result['experiment_params']['test_freq']}"

    fn =  f"benchmarks/results/{result['filename']}.yaml"

    if not os.path.exists(f"benchmarks/results/{result['lockdown_heuristic']}"):
        os.makedirs(f"benchmarks/results/{result['lockdown_heuristic']}")
    
    with open(fn, 'w') as file:
        yaml.dump(result, file)



def run_tree_heuristic(simulation_params, experiment_params, start_day, feature_set, tree_depth):
    ''' Takes a set of simulation_params and experiment parameters (delta_school, emotional cost of deaths (xi), max icus, max tests, testing and lockdown frequencies) and a set of simulation paramters (required by the constructor in group.py), creates a dynamical system, runs the tree heuristic and returns the dynamical system after running the heuristic. 
    '''

    start_time = time()

    # Define time variables
    num_time_periods = int(math.ceil(simulation_params["num_days"]/simulation_params["dt"]))


    # Read group parameters
    with open("parameters/"+simulation_params["region"]+".yaml","r") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        while file is None:
            print("Failed reading fitted param")
            file = open("parameters/"+simulation_params["region"]+".yaml","r")
        
        universe_params = yaml.load(file, Loader=yaml.FullLoader)
        
        while universe_params is None:
            print("Failed converting to yaml fitted param")
            universe_params = yaml.load(file, Loader=yaml.FullLoader)

        # Read initialization
    with open(f"initialization/{simulation_params['init']}.yaml","r") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        while file is None:
            print("Failed reading init param")
            file = open(f"initialization/{simulation_params['init']}.yaml","r")

        initialization = yaml.load(file, Loader=yaml.FullLoader)
        while initialization is None:
            print("Failed converting to yaml init param")
            initialization = yaml.load(file, Loader=yaml.FullLoader)
    
    # Read econ parameters
    with open(f"parameters/{simulation_params['econ']}.yaml","r") as file:
        while file is None:
            print("Failed reading econ param")
            file = open(f"initialization/{simulation_params['init']}.yaml","r")

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


    # Change eta
    dynModel.econ_params["employment_params"]["eta"] = simulation_params["eta"]

    tree_file = f"Dep_{tree_depth}_variables_{feature_set}.pickle"

    tree_policy_benchmark.run_tree_policy(dynModel, tree_file)

    end_time = time()

    print("Total running time for {} days is {}".format(simulation_params['num_days'], end_time - start_time))

    dynModel.print_stats()

    return dynModel



if __name__ == "__main__":
    main()
