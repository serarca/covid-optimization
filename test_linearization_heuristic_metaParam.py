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

from joblib import Parallel, delayed


def main():

    instance_index = 0

    if len(sys.argv) > 1:
        instance_index = int(sys.argv[1])

    # 30 * 37199.03
    # Some paramters to test the linearization heuristic
    scaling = 10000
    money_scaling = 1000
    
    params_to_try = {
        "delta_schooling":[0.5],
        "xi":[0, 30 * 37199.03 * scaling / money_scaling],
        # , 30 * 37199.03 * scaling / money_scaling],
        "icus":[3000 / scaling],
        "tests":[0, 30000 / scaling, 60000 / scaling],
        #  60000 / scaling],
        # , 30000 / scaling],
        "frequencies":[(1,1)],
        #  (7,14)],
        "region":["fitted-scaled"], 
        "econ": ["econ-scaled"],
        "init": ["60days-scaled"],
        "eta":[0, 0.1],
        "trust_region_radius":[0.05, 0.1, 0.2, 0.3],
        "max_inner_iterations_mult":[1, 1.5, 2, 3]
    }

    all_instances = list(it.product(*(params_to_try[param] for param in params_to_try)))

    n_days = 90
    groups = "all"
    start_day = 60

    scaling_econ_param(scaling, money_scaling)
    scaling_fitted(scaling, money_scaling)
    scaling_init(scaling)

    # Final time step is used if we want to evaluate 
    # the hueristic at any time before the n_days
    final_time_step = n_days
    
    # For names of regions see the "parameters" folder
    # region = 'fitted'

    delta = all_instances[instance_index][0]
    xi = all_instances[instance_index][1]
    icus = all_instances[instance_index][2]
    tests = all_instances[instance_index][3]
    test_freq = all_instances[instance_index][4][0]
    lockdown_freq = all_instances[instance_index][4][1]
    region = all_instances[instance_index][5]
    econ = all_instances[instance_index][6]
    init = all_instances[instance_index][7]
    eta = all_instances[instance_index][8]
    trust_region_radius = all_instances[instance_index][9]
    max_inner_iterations_mult = all_instances[instance_index][10]

    print(all_instances[instance_index])

    run_lin_heur_and_save_output(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, trust_region_radius, max_inner_iterations_mult)



def run_lin_heur_and_save_output(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, trust_region_radius, max_inner_iterations_mult):
    ''' Runs the linearization heuristic with the experiment parameters passed as arguments and saves the relevant output.'''

    experiment_params = {
        'delta_schooling':delta,
        'xi':xi,
        'icus':icus,
        'test_freq': test_freq,
        'lockdown_freq': lockdown_freq
    }

    simulation_params_linearization = {
        'dt':1.0,
        'region': region,
        'quar_freq': 1,
        'num_days' : n_days,
        'initial_infected_count' : 1,
        'mixing_method' : {
            "name":"mult"},
        'mtest_cap' : tests,
        'atest_cap' : tests,
        'heuristic': 'linearization',
        'transport_lb_work_fraction': 0.25,
        'econ': econ,
        'init': init,
        'eta': eta
    }

    total_reward, total_running_time = run_linearization_heuristic(simulation_params_linearization, experiment_params, start_day, trust_region_radius, max_inner_iterations_mult)

    with open(f"linearization_heur_meta_param_testing/testing_outputs_ndays={n_days}_eta={eta}_tests={tests}_xi={xi}.csv", "a+") as file:
        file.write(f"{trust_region_radius}, {max_inner_iterations_mult}, {max_inner_iterations_mult/trust_region_radius}, {total_reward}, {total_running_time} \n")
        file.close()



def run_linearization_heuristic(simulation_params, experiment_params, start_day, trust_region_radius, max_inner_iterations_mult):
    ''' Takes a set of simulation_params and experiment parameters (delta_school, emotional cost of deaths (xi), max icus, max tests, testing and lockdown frequencies) and a set of simulation paramters (required by the constructor in group.py), creates a dynamical system, runs the linearization heuristic and returns the dynamical system after running the heuristic. 
    '''

    start_time = time()

    # Define time variables
    num_time_periods = int(math.ceil(simulation_params["num_days"]/simulation_params["dt"]))


    # Read group parameters
    with open("parameters/"+simulation_params["region"]+".yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        universe_params = yaml.load(file, Loader=yaml.FullLoader)

        # Read initialization
    with open(f"initialization/{simulation_params['init']}.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        initialization = yaml.load(file, Loader=yaml.FullLoader)
        
    
    # Read econ parameters
    with open(f"parameters/{simulation_params['econ']}.yaml") as file:
        econ_params = yaml.load(file, Loader=yaml.FullLoader)

    

    # Define mixing method
    mixing_method = universe_params['mixing']

    eta = simulation_params["eta"]
    

    dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], num_time_periods, mixing_method, start_day, eta)

    # add parameters for testing capacity
    dynModel.parameters['global-parameters']['C_mtest'] = simulation_params['mtest_cap']
    dynModel.parameters['global-parameters']['C_atest'] = simulation_params['atest_cap']


    # Change eta
    dynModel.econ_params["employment_params"]["eta"] = simulation_params["eta"]


    linearization.run_heuristic_linearization(dynModel, trust_region_radius, max_inner_iterations_mult)

    end_time = time()

    total_running_time = end_time - start_time
    total_reward = dynModel.get_total_reward()

    print("Total running time for {} days is {}".format(simulation_params['num_days'], end_time - start_time))

    dynModel.print_stats()

    return total_reward, total_running_time

def scaling_econ_param(scaling, money_scaling):
    # Import data
    old_econ = yaml.load(open( "parameters/econ.yaml", "rb" ),Loader=yaml.FullLoader)
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


    with open('parameters/econ-scaled.yaml', 'w') as file:
        yaml.dump(scaled_econ, file)

def scaling_init(scaling):
    # Import data
    old_init = yaml.load(open( "initialization/60days.yaml", "rb" ), Loader=yaml.FullLoader)
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

    with open('initialization/60days-scaled.yaml', 'w') as file:
        yaml.dump(scaled_init_dict, file)


def scaling_fitted(scaling, money_scaling):
    # Import data
    old_fitted = yaml.load(open( "parameters/fitted.yaml", "rb" ), Loader=yaml.FullLoader)
    scaling = 1000.0

    scaled_fitted = dict(old_fitted)

    # Scale global_param
    scaled_fitted["global-parameters"]["C_H"] = scaled_fitted["global-parameters"]["C_H"] / scaling

    scaled_fitted["global-parameters"]["C_ICU"] = scaled_fitted["global-parameters"]["C_ICU"] / scaling



    for group_h in scaled_fitted["seir-groups"]:
        # # Scale contacts
        # for act in scaled_fitted["seir-groups"][group_h]["contacts"]:
        #     for group_g in scaled_fitted["seir-groups"][group_h]["contacts"][act]:
        #         scaled_fitted["seir-groups"][group_h]["contacts"][act][group_g] = scaled_fitted["seir-groups"][group_h]["contacts"][act][group_g] * scaling
        
        # Scale econ death value
        scaled_fitted["seir-groups"][group_h]["economics"]["death_value"] = scaled_fitted["seir-groups"][group_h]["economics"]["death_value"] * scaling
            

    with open('parameters/fitted-scaled.yaml', 'w') as file:
        yaml.dump(scaled_fitted, file)



if __name__ == "__main__":
    main()
