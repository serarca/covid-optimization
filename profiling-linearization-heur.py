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

from group import SEIR_group, DynamicalModel
from heuristics import *
import linearization
# from forecasting_heuristic import *
import math
import pprint
from time import time

import pickle
import pandas as pd
import logging

from joblib import Parallel, delayed

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

# @log_execution_time
def run_linearization_heuristic(simulation_params, experiment_params):

    start_time = time()

    # Define time variables
    num_time_periods = int(math.ceil(simulation_params["num_days"]/simulation_params["dt"]))

    # Define mixing method
    mixing_method = simulation_params['mixing_method']

    # Read group parameters
    with open("parameters/"+simulation_params["region"]+".yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        universe_params = yaml.load(file, Loader=yaml.FullLoader)

        # Read initialization
    with open("initialization/fitted.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        initialization = yaml.load(file, Loader=yaml.FullLoader)
    # Read econ parameters

    with open("parameters/econ.yaml") as file:
        econ_params = yaml.load(file, Loader=yaml.FullLoader)

    # Percentage infected at time 0
    perc_infected = simulation_params['perc_infected']
    # Move population to infected (without this there is no epidem.)
    for group in initialization:
        change = initialization[group]["S"]*perc_infected/100
        initialization[group]["S"] = initialization[group]["S"] - change
        initialization[group]["I"] = initialization[group]["I"] + change

    dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], num_time_periods, mixing_method, simulation_params['transport_lb_work_fraction'])

    # add parameters for testing capacity
    dynModel.parameters['global-parameters']['C_mtest'] = simulation_params['mtest_cap']
    dynModel.parameters['global-parameters']['C_atest'] = simulation_params['atest_cap']


    linearization.run_heuristic_linearization(dynModel)


    end_time = time()

    print("Total running time for {} days is {}".format(simulation_params['num_days'], end_time - start_time))

    return dynModel



def main():

    # logging.basicConfig(
    #     level=logging.CRITICAL,
    #     filename=f'profiling-linearized-heuristic/profiling-logging.txt',
    #     format='%(message)s')

    params_to_try = {
        "delta_schooling":[0.5],
        "xi":[1 * 37199.03, 30 * 37199.03],
        "icus":[2000,2500],
        "tests":[0,60000],
    }
    regions = ['fitted']
    # 'testing_5_groups']
    # 'Testing-group', 'Ile-de-France']
    n_days = 180
    final_time_step = 90
    region = 'fitted'

    Parallel(n_jobs=8)(delayed(run_lin_heur_and_pickle_dynModel)(delta, xi, icus, tests, n_days, region)
    for delta in params_to_try["delta_schooling"]
    for xi in params_to_try["xi"]
    for icus in params_to_try["icus"]
    for tests in params_to_try["tests"])

    load_pickles_and_create_csv(n_days, params_to_try, final_time_step)

def run_lin_heur_and_pickle_dynModel(delta, xi, icus, tests, n_days, region):

    experiment_params = {
        'delta_schooling':delta,
        'xi':xi,
        'icus':icus,
    }
    # logging.critical(f'{region}, {n_days}')

    simulation_params_linearization = {
        'dt':1.0,
        'region': region,
        'quar_freq': 1,
        'num_days' : n_days,
        'initial_infected_count' : 1,
        'perc_infected' : 10,
        'mixing_method' : {
            "name":"mult",
            "param_alpha":1.0,
            "param_beta":0.5,},
        'mtest_cap' : 100,
        'atest_cap' : 100,
        'work_full_lockdown_factor' : 0.24,
        'heuristic': 'linearization',
        'transport_lb_work_fraction': 0.25
    }

    dynModel_linearization_heur = run_linearization_heuristic(simulation_params_linearization, experiment_params)
    # logging.critical('*')

    pickle.dump(dynModel_linearization_heur,open(f"linearization_heuristic_dyn_models/dynModel_linHeur_n_days={n_days}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}.p","wb"))


def load_pickles_and_create_csv(n_days, params_to_try, final_time_step):
    results = []
    for delta in params_to_try["delta_schooling"]:
        for xi in params_to_try["xi"]:
            for icus in params_to_try["icus"]:
                for tests in params_to_try["tests"]:

                    dynModel = pickle.load(open(f"linearization_heuristic_dyn_models/dynModel_linHeur_n_days={n_days}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}.p","rb"))

                    results.append({
                        "heuristic":"linearization_heuristic",
                        "delta_schooling":delta,
                        "xi":xi,
                        "icus":icus,
                        "tests":tests,
                        "testing":"linearization_heuristic",
                        "economics_value":dynModel.get_total_economic_value(final_time_step),
                        "deaths":dynModel.get_total_deaths(final_time_step),
                        "reward":dynModel.get_total_reward(final_time_step),
                    })

    pd.DataFrame(results).to_csv("linearization_heuristic_dyn_models/linearization_heuristic_results.csv")





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
    fig = plt.figure()
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
