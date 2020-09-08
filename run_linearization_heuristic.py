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

import pickle
import pandas as pd
import logging

from joblib import Parallel, delayed


def main():

    # Some paramters to test the linearization heuristic
    scaling = 10000
    money_scaling = 1
    # params_to_try = {
    #     "delta_schooling":[0.5],
    #     "xi":[30 * 37199.03 * scaling / money_scaling],
    #     "icus":[3000 / scaling],
    #     "tests":[0/ scaling],
    #     "frequencies":[(1,1)],
    #     "region":["one_group_fitted-scaled-test"], 
    #     "econ": ["one_group_econ-scaled"],
    #     "init": ["60days_one_group-scaled"],
    #     "eta":[0.1]
    # }
    params_to_try = {
        "delta_schooling":[0.5],
        "xi":[30 * 37199.03],
        "icus":[3000],
        "tests":[0],
        "frequencies":[(1,1)],
        "region":["one_group_fitted"], 
        "econ": ["one_group_econ-zeroEconVal"],
        "init": ["60days_one_group"],
        "eta":[0.1]
    }

    n_days = 50
    
    # Final time step is used if we want to evaluate 
    # the hueristic at any time before the n_days
    final_time_step = n_days
    
    # For names of regions see the "parameters" folder
    # region = 'fitted'
    
    
    Parallel(n_jobs=4)(delayed(run_lin_heur_and_pickle_dynModel)(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta)
    for delta in params_to_try["delta_schooling"]
    for xi in params_to_try["xi"]
    for icus in params_to_try["icus"]
    for tests in params_to_try["tests"]
    for test_freq, lockdown_freq in params_to_try['frequencies']
    for econ in params_to_try['econ']
    for init in params_to_try['init']
    for region in params_to_try['region']
    for eta in params_to_try['eta'])

    # for delta in params_to_try["delta_schooling"]:
    #     for xi in params_to_try["xi"]:
    #         for icus in params_to_try["icus"]:
    #             for tests in params_to_try["tests"]:
    #                 for test_freq, lockdown_freq in params_to_try['frequencies']:
    #                     for econ in params_to_try['econ']:
    #                         for init in params_to_try['init']:
    #                             for region in params_to_try['region']:

    #                                 run_lin_heur_and_pickle_dynModel(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta)

    run_all_pickled_dynModels_prop_bouncing(n_days, params_to_try)

    # unpickle_plot_and_print_results(n_days, params_to_try, simulation_params_linearization)

    load_pickles_and_create_csv(n_days, params_to_try, final_time_step)

def run_lin_heur_and_pickle_dynModel(delta, xi, icus, tests, n_days, region, test_freq, lockdown_freq, econ, init, eta):
    ''' Runs the linearization heuristic with the experiment parameters passed as arguments and saves the resulting dynamical model as a pickle object.'''

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

    dynModel_linearization_heur = run_linearization_heuristic(simulation_params_linearization, experiment_params)

    pickle.dump(dynModel_linearization_heur,open(f"linearization_heuristic_dyn_models/dynModel_linHeur_n_days={n_days}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}_testFreq={test_freq}_lockFreq={lockdown_freq}_eta={eta}.p","wb"))


def run_linearization_heuristic(simulation_params, experiment_params):
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
        start_day = 60
    
    # Read econ parameters
    with open(f"parameters/{simulation_params['econ']}.yaml") as file:
        econ_params = yaml.load(file, Loader=yaml.FullLoader)

    

    # Define mixing method
    mixing_method = universe_params['mixing']
    

    dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], num_time_periods, mixing_method, start_day)

    # add parameters for testing capacity
    dynModel.parameters['global-parameters']['C_mtest'] = simulation_params['mtest_cap']
    dynModel.parameters['global-parameters']['C_atest'] = simulation_params['atest_cap']


    # Change eta
    dynModel.econ_params["employment_params"]["eta"] = simulation_params["eta"]


    linearization.run_heuristic_linearization(dynModel)

    end_time = time()

    print("Total running time for {} days is {}".format(simulation_params['num_days'], end_time - start_time))

    dynModel.print_stats()

    return dynModel


def run_all_pickled_dynModels_prop_bouncing(n_days, params_to_try):
    ''' Loads all the pickled dynamical models and runs them with proportional bouncing, saving the results in another pickle object'''

    for delta in params_to_try["delta_schooling"]:
        for xi in params_to_try["xi"]:
            for icus in params_to_try["icus"]:
                for tests in params_to_try["tests"]:
                    for test_freq, lockdown_freq in params_to_try['frequencies']:
                        for eta in params_to_try["eta"]:
                            pickled_dyn_model = f"linearization_heuristic_dyn_models/dynModel_linHeur_n_days={n_days}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}_testFreq={test_freq}_lockFreq={lockdown_freq}_eta={eta}.p"

                            run_dyn_model_with_no_bouncing_and_pickle(pickled_dyn_model)


def run_dyn_model_with_no_bouncing_and_pickle(pickled_dyn_model):
    ''' Loads a pickled dynamical model and re-runs it with proportional-bouncing, then saves the resulting dynamical model into another pickle object'''

    dynModel = pickle.load(open(pickled_dyn_model,"rb"))

    n_days = int(dynModel.time_steps * dynModel.dt)
    delta = dynModel.experiment_params['delta_schooling']
    xi = dynModel.experiment_params['xi']
    icus = dynModel.icus
    tests = dynModel.parameters['global-parameters']['C_mtest']

    # Recover all the controls from the 
    # dynamical model except for bouncing decisions
    lockdowns = dynModel.lockdown_controls
    m_tests_cont = dynModel.m_tests_controls
    a_tests_cont = dynModel.a_tests_controls

    dynModel.reset_time(0)
    dynModel.simulate(m_tests_cont, a_tests_cont, lockdowns)

    pickle.dump(dynModel,open(f"linearization_heuristic_dyn_models/dynModel_linHeur_Prop_Bouncing_n_days={n_days}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}_testFreq={dynModel.experiment_params['test_freq']}_lockFreq={dynModel.experiment_params['lockdown_freq']}_eta={dynModel.econ_params['employment_params']['eta']}.p","wb"))


def load_pickles_and_create_csv(n_days, params_to_try, final_time_step):
    ''' Loads all pickled dynamical models and creates an excel spreadsheet to visualize the resulting metrics.'''

    results = []
    for delta in params_to_try["delta_schooling"]:
        for xi in params_to_try["xi"]:
            for icus in params_to_try["icus"]:
                for tests in params_to_try["tests"]:
                    for heur in ["","_Prop_Bouncing"]:
                        for test_freq, lockdown_freq in params_to_try['frequencies']:
                            for eta in params_to_try["eta"]:
                            
                                dynModel = pickle.load(open(f"linearization_heuristic_dyn_models/dynModel_linHeur{heur}_n_days={n_days}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}_testFreq={test_freq}_lockFreq={lockdown_freq}_eta={eta}.p","rb"))

                                results.append({
                                    "heuristic":f"linearization_heuristic{heur}",
                                "delta_schooling":delta,
                                "lockdown_freq":lockdown_freq,
                                "test_freq":test_freq,
                                "xi":xi,
                                "icus":icus,
                                "tests":tests,
                                "eta":eta,
                                "testing":"linearization_heuristic",
                                "economics_value":dynModel.get_total_economic_value(final_time_step),
                                "deaths":dynModel.get_total_deaths(final_time_step),
                                "reward":dynModel.get_total_reward(final_time_step)
                            })

    pd.DataFrame(results).to_excel(f"linearization_heuristic_dyn_models/linearization_heuristic_results_{n_days}_days.xlsx")


def unpickle_plot_and_print_results(n_days, params_to_try, simulation_params):
    '''PLOT AND PRINT RESULTS FROM THE DYNAMICAL MODEL SIMULATED USING THE SIMULATION PARAMS'''

    for delta in params_to_try["delta_schooling"]:
        for xi in params_to_try["xi"]:
            for icus in params_to_try["icus"]:
                for tests in params_to_try["tests"]:
                    for heur in ["","_Prop_Bouncing"]:
                        for test_freq, lockdown_freq in params_to_try['frequencies']:
                            for eta in params_to_try["eta"]:

                                dynModel = pickle.load(open(f"linearization_heuristic_dyn_models/dynModel_linHeur{heur}_n_days={n_days}_deltas={delta}_xi={xi}_icus={icus}_maxTests={tests}_testFreq={test_freq}_lockFreq={lockdown_freq}_eta={eta}.p","rb"))

                                # heuristic = simulation_params['heuristic']+heur
                                T = dynModel.time_steps

                                K_mtest = tests
                                K_atest = tests

                                # Retrieve optimal lockdown decisions
                                # Express as dictionary where given an age group, an activity key corresponds to an np.array of length T.
                                # That array holds the lockdown decisions for that age group and that activity used in the simulation of dynModel.
                                lockdowns_sim = {}
                                for n in dynModel.groups:
                                    lockdowns_sim[n] = {}
                                    for act in dynModel.lockdown_controls[0][n]:
                                        lockdowns_sim[n][act] = np.zeros(T)
                                        for t in range(T):
                                            lockdowns_sim[n][act][t] = dynModel.lockdown_controls[t][n][act]

                                # Retrieve simulated testing decisions
                                m_tests_sim = {}
                                for n in dynModel.groups:
                                    m_tests_sim[n] = np.zeros(T)
                                    for t in range(T):
                                        m_tests_sim[n][t] = dynModel.m_tests_controls[t][n]

                                a_tests_sim = {}
                                for n in dynModel.groups:
                                    a_tests_sim[n] = np.zeros(T)
                                    for t in range(T):
                                        a_tests_sim[n][t] = dynModel.a_tests_controls[t][n]


                                # # Retrieve simulated bouncing decisions

                                B_H_sim = {}
                                for n,g in dynModel.groups.items():
                                    B_H_sim[n] = np.zeros(T-1)
                                    for t in range(T-1):
                                        B_H_sim[n][t] = g.B_H[t]

                                B_ICU_sim = {}
                                for n,g in dynModel.groups.items():
                                    B_ICU_sim[n] = np.zeros(T-1)
                                    for t in range(T-1):
                                        B_ICU_sim[n][t] = g.B_ICU[t]

                                print("Deaths through the end of the horizon:", sum([dynModel.deaths[t] for t in range(0,T+1) if t!=0]))
                                print("Economic output through the end of the horizon:", sum([dynModel.economic_values[t] for t in range(0,T+1) if t!=0]))



                                # Plotting
                                time_axis = [i*simulation_params["dt"] for i in range(T+1)]
                                time_axis_controls = [i*simulation_params["dt"] for i in range(T)]

                                groups = dynModel.groups.keys()
                                groups = sorted(groups)
                                
                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1)
                                    plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
                                    plt.title(group)
                                    plt.legend(loc='upper right')
                                    plt.ylim(-1,np.max([np.max(dynModel.groups[group].S) for group in groups]))

                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1+len(groups))
                                    plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
                                    plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
                                    plt.legend(loc='upper right')
                                    plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].E),np.max(dynModel.groups[group].I)) for group in groups]))

                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1+len(groups)*2)
                                    plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
                                    plt.ylim(-1,np.max([np.max(dynModel.groups[group].R) for group in groups]))
                                    plt.legend(loc='upper right')

                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1+len(groups)*3)
                                    plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
                                    plt.ylim(-1,np.max([np.max(dynModel.groups[group].Rq) for group in groups]))
                                    plt.legend(loc='upper right')

                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1+len(groups)*4)
                                    plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
                                    plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
                                    plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
                                    plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
                                    plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].Ia),np.max(dynModel.groups[group].Ips),np.max(dynModel.groups[group].Ims),np.max(dynModel.groups[group].Iss)) for group in groups]))
                                    plt.legend(loc='upper right')

                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1+len(groups)*5)
                                    plt.plot(time_axis, dynModel.groups[group].H, label="Hospital")
                                    plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
                                    plt.plot(time_axis, dynModel.groups[group].D, label="Dead")
                                    plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].H),np.max(dynModel.groups[group].ICU),np.max(dynModel.groups[group].D)) for group in groups]))
                                    plt.legend(loc='upper right')


                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1+len(groups)*6)
                                    plt.plot(time_axis_controls,m_tests_sim[group], label = "M tests")
                                    plt.plot(time_axis_controls,a_tests_sim[group], label = "A tests")
                                    # plt.plot(range(0,int(simulation_params['time_periods'])),
                                    # np.array(re_change_order(m_tests_vec)[group])+max(float(args.m_tests),float(args.a_tests))/100, label="M Tests")
                                    # plt.plot(range(0,int(simulation_params['time_periods'])), re_change_order(a_tests_vec)[group], label="A Tests")
                                    plt.ylim(-max(float(K_mtest),float(K_atest))/10,max(float(K_mtest),float(K_atest))+max(float(K_mtest),float(K_atest))/10)
                                    plt.legend(loc='upper right')

                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1+len(groups)*7)
                                    plt.plot(time_axis_controls, lockdowns_sim[group]["home"]+0.01, label="Home")
                                    plt.plot(time_axis_controls, lockdowns_sim[group]["work"]+0.01*2, label="Work")
                                    plt.plot(time_axis_controls, lockdowns_sim[group]["transport"]+0.01*3, label="Transport")
                                    plt.plot(time_axis_controls, lockdowns_sim[group]["school"]+0.01*4, label="School")
                                    plt.plot(time_axis_controls, lockdowns_sim[group]["leisure"]+0.01*5, label="Leisure")
                                    plt.plot(time_axis_controls, lockdowns_sim[group]["other"]+0.01*6, label="Other")
                                    plt.ylim(-0.1,1.1)
                                    plt.legend(loc='upper right')

                                for i,group in enumerate(groups):
                                    plt.subplot(13,len(groups),i+1+len(groups)*8)
                                    plt.plot(time_axis_controls, dynModel.groups[group].B_H, label="Bounced H")
                                    plt.plot(time_axis_controls, dynModel.groups[group].B_ICU, label="Bounced ICU")
                                    plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].B_H),np.max(dynModel.groups[group].B_ICU)) for group in groups]))
                                    plt.legend(loc='upper right')

                                plt.subplot(13,2,19)
                                #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
                                plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
                                #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
                                plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
                                plt.legend(loc='upper right')

                                plt.subplot(13,2,20)
                                #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
                                plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
                                #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
                                plt.legend(loc='upper right')

                                figure = plt.gcf() # get current figure
                                figure.set_size_inches(7*len(groups),24)
                                figure.suptitle('Region: %s, %s Heuristic with Total Days: %s, Initial Infected percentage: %2d, M-test daily capacity: %s, A-test daily capacity: %s, Mixing: %s'%(simulation_params['region'],simulation_params['heuristic'],T,simulation_params['perc_infected'],K_mtest,K_atest,simulation_params['mixing_method']["name"]), fontsize=22)
                                plt.savefig("results_runs/linearization_heuristic_dyn_models/"+simulation_params['region']+"_"+simulation_params['heuristic']+heur+"_heuristic"+"_n_days_"+str(T)+"_tests_"+str(dynModel.parameters['global-parameters']['C_mtest'])+ "_icu_cap_"+str(dynModel.icus)+"_deltaS_"+str(delta)+"_xi_"+str(xi)+"_mixing_" + simulation_params['mixing_method']["name"]+"_testFreq="+str(test_freq)+"_lockFreq="+str(lockdown_freq)+".pdf")
                                plt.close()







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
