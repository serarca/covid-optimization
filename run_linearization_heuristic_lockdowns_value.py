# -*- coding: utf-8 -*-
import yaml
from inspect import getsourcefile
import os.path
import os
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

#from joblib import Parallel, delayed


def main():

    instance_index = 0

    if len(sys.argv) > 1:
        instance_index = int(sys.argv[1])

    # 30 * 37199.03
    # Some paramters to test the linearization heuristic
    scaling = 10000
    money_scaling = 1000
    xi_mult_values = [50]
    # total_population = 12278209.99439713
    # testing_values_perc = np.linspace(0,0.035,num=100)

    # testing_values = [total_population * p for p in testing_values_perc]

    # , 60000, 120000]
    icu_values = [2900]
    # , 2300, 2600, 2900, 3200]

    params_to_try = {
        "delta_schooling":[0.5],
        # , 1, 5],
        "icus":[ic / scaling for ic in icu_values],
        "mtests":[0],
        # "atests":[test_cap / scaling for test_cap in testing_values],
        "frequencies":[(7,14)],
        "region":["fitted-scaled"], 
        "econ": ["econ-scaled"],
        "init": ["oct21-scaled"],
        "eta":[0.1],
        "xi":[mult * 37199.03 * scaling / money_scaling for mult in xi_mult_values],
        "trust_region_radius":[0.05],
        "max_inner_iterations_mult":[2],
        "initial_uhat":["full_open"],
        "pLock":np.linspace(0,1,num=100)
        # , "dynamic_gradient"]
        # "full_lockdown", "full_open","dynamic_gradient", "activity_gradient", "age_group_gradient", "time_gradient"
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


    scaling_econ_param(scaling, money_scaling, groups)
    scaling_fitted(scaling, money_scaling, groups)
    scaling_init(scaling, groups)

    # Final time step is used if we want to evaluate 
    # the hueristic at any time before the n_days
    final_time_step = n_days
    
    # For names of regions see the "parameters" folder
    # region = 'fitted'

    delta = all_instances[instance_index][0]
    icus = all_instances[instance_index][1]
    mtests = all_instances[instance_index][2]
    atests = 0

    # atests = all_instances[instance_index][4]
    print(all_instances[instance_index])
    test_freq = all_instances[instance_index][3][0]
    lockdown_freq = all_instances[instance_index][3][1]
    region = all_instances[instance_index][4]
    econ = all_instances[instance_index][5]
    init = all_instances[instance_index][6]
    eta = all_instances[instance_index][7]
    xi = all_instances[instance_index][8]
    trust_region_radius = all_instances[instance_index][9]
    max_inner_iterations_mult = all_instances[instance_index][10]
    initial_uhat = all_instances[instance_index][11]
    pLock = all_instances[instance_index][12]
    
    targetActivities = False
    targetGroups = True
    
    optimizeLockdowns = True
    averageLockConst = True
    targetTests = True

    # if initial_uhat == "time_gradient":
    #     targetActivities = False
    #     targetGroups = False
    
    # if initial_uhat == "age_group_gradient":
    #     targetActivities = False
    
    # if initial_uhat == "activity_gradient":
    #     targetGroups = False


    run_lin_heur_and_save_yaml(delta, xi, icus, mtests, atests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, scaling, money_scaling, targetActivities, targetGroups, targetTests, optimizeLockdowns, averageLockConst, pLock)
    
    
    # run_lin_heur_and_pickle_dynModel(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, targetActivities, targetGroups, targetTests)

    # run_pickled_dynModels_prop_bouncing(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day)

    # for heur in ["", "_Prop_Bouncing"]:
    #     load_pickle_and_create_yaml(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, scaling, money_scaling, heur)


def run_lin_heur_and_save_yaml(delta, xi, icus, mtests, atests, n_days, region, test_freq, lockdown_freq, econ, init, eta, groups, start_day, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, scaling, money_scaling, targetActivities, targetGroups, targetTests, optimizeLockdowns, averageLockConst, pLock):
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

    dynModel = run_linearization_heuristic(simulation_params, experiment_params, start_day, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, targetActivities, targetGroups, targetTests, optimizeLockdowns, averageLockConst, pLock)

    result = {
            "lockdown_heuristic":f"linearization_heuristic_optBouncing={optimize_bouncing}_initial_uhat={initial_uhat}_targetGroups={targetGroups}_targetAct={targetActivities}_targetTests={targetTests}_optimizeLockdowns={optimizeLockdowns}_lockdownAvgConst={averageLockConst}",
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

    result["filename"] = f"{result['lockdown_heuristic']}/xi-{result['experiment_params']['xi']}_icus-{result['experiment_params']['icus']}_testing-{result['testing_heuristic']}_natests-{result['experiment_params']['n_a_tests']}_nmtests-{result['experiment_params']['n_m_tests']}_T-{result['experiment_params']['T']}_startday-{result['experiment_params']['start_day']}_groups-{result['groups']}_dschool-{result['experiment_params']['delta_schooling']}_eta-{result['experiment_params']['eta']}_lockdownFreq-{result['experiment_params']['lockdown_freq']}_testingFreq-{result['experiment_params']['test_freq']}_plock={pLock}"

    fn =  f"benchmarks/results/{result['filename']}.yaml"
    if not os.path.exists(f"benchmarks/results/{result['lockdown_heuristic']}"):
        os.makedirs(f"benchmarks/results/{result['lockdown_heuristic']}")

    with open(fn, 'w') as file:
        yaml.dump(result, file)





def run_linearization_heuristic(simulation_params, experiment_params, start_day, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, targetActivities, targetGroups, targetTests, optimizeLockdowns, averageLockConst, pLock):
    ''' Takes a set of simulation_params and experiment parameters (delta_school, emotional cost of deaths (xi), max icus, max tests, testing and lockdown frequencies) and a set of simulation paramters (required by the constructor in group.py), creates a dynamical system, runs the linearization heuristic and returns the dynamical system after running the heuristic. 
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


    linearization.run_heuristic_linearization(dynModel, trust_region_radius, max_inner_iterations_mult, initial_uhat, optimize_bouncing, targetActivities, targetGroups, targetTests, False, 0.1, optimizeLockdowns,averageLockConst, pLock)

    end_time = time()

    print("Total running time for {} days is {}".format(simulation_params['num_days'], end_time - start_time))

    dynModel.print_stats()

    return dynModel




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




def scaling_econ_param(scaling, money_scaling, groups):
    # Import data
    if groups == "all":
        old_econ = yaml.load(open( "parameters/econ.yaml", "rb" ),Loader=yaml.FullLoader)
    elif groups == "one":
        old_econ = yaml.load(open( "parameters/one_group_econ.yaml", "rb" ),Loader=yaml.FullLoader)
    
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
        with open('parameters/econ-scaled.yaml', 'w') as file:
            yaml.dump(scaled_econ, file)
    elif groups == "one":
        with open('parameters/one_group_econ-scaled.yaml', 'w') as file:
            yaml.dump(scaled_econ, file)
    
    

def scaling_init(scaling, groups):
    # Import data
    if groups == "all":
        old_init = yaml.load(open( "initialization/oct21.yaml", "rb" ), Loader=yaml.FullLoader)
    elif groups == "one":
        old_init = yaml.load(open( "initialization/oct21_one_group.yaml", "rb" ), Loader=yaml.FullLoader)

    
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
        with open('initialization/oct21-scaled.yaml', 'w') as file:
            yaml.dump(scaled_init_dict, file)
    elif groups == "one":
        with open('initialization/oct21_one_group-scaled.yaml', 'w') as file:
            yaml.dump(scaled_init_dict, file)

    


def scaling_fitted(scaling, money_scaling, groups):

    if groups == "all":
        # Import data
        old_fitted = yaml.load(open( "parameters/fitted.yaml", "rb" ), Loader=yaml.FullLoader)
    elif groups == "one":
        # Import data
        old_fitted = yaml.load(open( "parameters/one_group_fitted.yaml", "rb" ), Loader=yaml.FullLoader)

    # scaling = 1000.0

    scaled_fitted = dict(old_fitted)

    # Scale global_param
    scaled_fitted["global-parameters"]["C_H"] = scaled_fitted["global-parameters"]["C_H"] / scaling

    scaled_fitted["global-parameters"]["C_ICU"] = scaled_fitted["global-parameters"]["C_ICU"] / scaling



    # for group_h in scaled_fitted["seir-groups"]:
        # # Scale contacts
        # for act in scaled_fitted["seir-groups"][group_h]["contacts"]:
        #     for group_g in scaled_fitted["seir-groups"][group_h]["contacts"][act]:
        #         scaled_fitted["seir-groups"][group_h]["contacts"][act][group_g] = scaled_fitted["seir-groups"][group_h]["contacts"][act][group_g] * scaling
        
        # Scale econ death value
        # scaled_fitted["seir-groups"][group_h]["economics"]["death_value"] = scaled_fitted["seir-groups"][group_h]["economics"]["death_value"] * scaling
            

    if groups == "all":
        with open('parameters/fitted-scaled.yaml', 'w') as file:
            yaml.dump(scaled_fitted, file)
    elif groups == "one":
        with open('parameters/one_group_fitted-scaled.yaml', 'w') as file:
            yaml.dump(scaled_fitted, file)



def plot_logging(file):
    number_of_groups = []
    n_days = []

    function_total_times = {}

    with open(file, 'r+') as file:
        line = file.readline().strip().split(",")
        while line:

            if line[0] == "*":
                line = file.readline().strip().split(",")
                if line == ['']:
                    break

            first_lines = line
            if first_lines[0] not in number_of_groups:
                number_of_groups.append(first_lines[0])

            if first_lines[1] not in n_days:
                n_days.append(int(first_lines[1]))

            if first_lines[0] not in function_total_times:
                function_total_times[first_lines[0]] = {}

            if int(first_lines[1]) not in function_total_times[first_lines[0]]:
                function_total_times[first_lines[0]][int(first_lines[1])] = {}

            line = file.readline().strip().split(",")
            while line[0] != "*":
                if line[0] not in function_total_times[first_lines[0]][int(first_lines[1])]:
                    function_total_times[first_lines[0]][int(first_lines[1])][line[0]] = float(line[1])
                else:
                    function_total_times[first_lines[0]][int(first_lines[1])][line[0]] += float(line[1])
                line = file.readline().strip().split(",")

    print(function_total_times)

    time_axis = sorted(n_days)
    # fig = plt.figure()
    plt.subplot(111)
    print(number_of_groups)

    for n_group in number_of_groups:
        print(function_total_times[n_group][time_axis[0]])
        for func in function_total_times[n_group][time_axis[0]]:

            plt.plot(time_axis, [function_total_times[n_group][t][func] for t in time_axis], label=f"Function: {func} Number of Groups: {n_group}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel("time (in sec)")
    plt.xlabel("Total time horizon T used")

    plt.savefig('profiling-linearized-heuristic/profiling-lin-heur.pdf',bbox_inches="tight")



if __name__ == "__main__":
    main()
