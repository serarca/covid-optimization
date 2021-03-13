# %matplotlib inline
import yaml
from inspect import getsourcefile
import os.path
import os
os.system('module load gurobipy3')

import sys
import matplotlib
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from io import StringIO
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import datetime as DT

import seaborn as sns
# import gurobipy
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_parent_dir+"/heuristics")

print(parent_parent_dir+"/heuristics")

# parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_parent_dir)


from group import SEIR_group, DynamicalModel
import math
import pprint
import pandas as pd
import pickle
import numpy as np
from copy import deepcopy



#### COMPUTE V

T = 104

experiment_params = {
  "T":90,
    "delta_schooling":0.5,
    "eta":0.1,
    "icus":2900.0,
    "lockdown_freq":14,
    "n_a_tests":0,
    "n_m_tests":60000.0,
    "start_day":0,
    "test_freq":7,
    'xi':0.0
    }

# Read group parameters
with open("../../parameters/fitted.yaml") as file:
    universe_params_all = yaml.load(file, Loader=yaml.FullLoader)

# Read econ parameters
with open("../../parameters/econ.yaml") as file:
    econ_params_all = yaml.load(file, Loader=yaml.FullLoader)

start_day = experiment_params["start_day"]


universe_params = universe_params_all
econ_params = econ_params_all

with open("../../initialization/oct21.yaml") as file:
    initialization = yaml.load(file, Loader=yaml.FullLoader)

experiment_params = experiment_params

dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, 1, experiment_params["T"], universe_params["mixing"], start_day, experiment_params['eta'], extra_data = True)



econ_activities = ["transport","leisure","other"]

V = 0

for age_group in dynModel.age_groups:
    total_initial_population_g = sum([dynModel.initialization[age_group][cat] for cat in ["N","E","I","ICU","Ia","Ims","Ips","Iss","Rq"] ])

#   D: 2.0129755906530393
#   E: 2285.276888763268
#   H: 13.540885275895503
#   I: 2142.166336075084
#   ICU: 3.5669812156992835
#   Ia: 0.0
#   Ims: 0.0
#   Ips: 0.0
#   Iss: 0.0
#   N: 1594534.826129878
#   R: 160265.76119148443
#   Rq: 302.0529959235804
#   S: 1429841.621713554
    # print(f"age group: {age_group}")
    # print(f"Initial Pop: {total_initial_population_g}")
    v_g = sum([dynModel.econ_params["employment_params"]["v"][age_group][activity] for activity in econ_activities])
    
    v_employment_1_g = v_g

    # print(f"employment: {v_g * 365}")

    v_schooling_1_g =  (

        dynModel.experiment_params['delta_schooling']*dynModel.econ_params['schooling_params'][age_group]
    )


    V += (v_employment_1_g + v_schooling_1_g) * T * total_initial_population_g

print(V)


#########
# Loss is V - reward 
##########








init_param = ["alpha_mixing", "multiplier_beta", "multiplier_p_icu", "multiplier_p_d","multiplier_lambda_h", "multiplier_lambda_icu", "alpha_other","l_school_march", "l_school_may", "l_school_july", "l_school_september",  "econ_value", "l_work_april"]

xi_mult_values = [0,10,25,50,100,150]

testing_values = [0]
   
icu_values = [2900]

random_instances=range(0,40)

T = 90
e = 0.1
d = 0.5
i = 2900
lockdown_freq = 14
testing_freq = 7
nm = 0.0
na =0

xi_to_try = [mult * 37199.03  for mult in xi_mult_values]

initial_uhats = ["time_gradient","age_group_gradient", "activity_gradient", "dynamic_gradient"]
# "time_gradient"
# "age_group_gradient", "activity_gradient", "dynamic_gradient"

# "linearization_heuristic_optBouncing=False_initial_uhat="



# age_group_gradient_"

try:
    old_data = pd.read_excel("plots/results-Robustness.xlsx")
except FileNotFoundError:
    old_data = pd.DataFrame([{
                                            "init_u_hat":" ",
                                            "xi":0,
                                            "icus":0,
                                            "n_m_tests":0,
                                            "eta":0,
                                            "init_param":" ",
                                            "randomInstance":-1,
                                            "economics_value":0,
                                            "deaths":0,
                                            "reward":0,
                                            "economic_losses":0,
                                            "total_loss":0
                                        }])


old_data = old_data.drop_duplicates()

new_data = []


for u_hat in initial_uhats:
    for param in init_param:
        
        h = f"linearization_heuristic_optBouncing=False_initial_uhat={u_hat}_initParamVarying={param}"
                        
                       
                        
        for file_name in os.listdir(h):
            instance = file_name.split("_")

            for p in instance:
                if "xi" in p and "mixing" not in p:
                    xi = float(p.split("-")[1])
    #                 print(xi)
                # if "icus" in p:
                #     icu = float(p.split("-")[1])
    #                 print(icu)
                if "natests" in p:
                    natest = float(p.split("-")[1])
    #                 print(natest)
                if "nmtests" in p:
                    nmtest = float(p.split("-")[1])
    #                 print(nmtest)
                if "dschool" in p:
                    delta = float(p.split("-")[1])
    #                 print(delta)
                # if "eta" in p:
                #     eta = float(p.split("-")[1])
    #                 print(eta)
                if "T" in p:
                    Time = float(p.split("-")[1])
    #                 print(eta)
                if "randomInstance" in p:
                    randomInstance = int(p.split("-")[1][:-5])
                

            if   nmtest == nm and natest == na and delta == d and Time==T:
                

                if old_data.loc[(old_data["randomInstance"] == randomInstance) & (old_data["init_u_hat"] == u_hat) & (old_data["init_param"] == param) & (old_data["xi"] == round(xi/37199.03))].empty:

                    
                    # # print("Found in results")
                    # deaths = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["deaths"].item()
                    # econ_value = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["economic_losses"].item()
                    # reward = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["total_loss"].item()
                    # print(f"reward is {reward}")
                    
                    # print(f"Econ value rerun: {econ_value}")
                    # with open("%s/%s"%(h,file_name)) as file:
                    #     result = yaml.load(file, Loader=yaml.UnsafeLoader)
                    # printf(f"Econ value original: {result['results']['econ_value']/10e9}")



                    with open("%s/%s"%(h,file_name)) as file:
                        result = yaml.load(file, Loader=yaml.UnsafeLoader)


                    

                    # Read group parameters
                    with open("../../parameters/simulations/fitted"+f"_{init_param.index(param)}_{randomInstance}.yaml","r") as file:
                        # The FullLoader parameter handles the conversion from YAML
                        # scalar values to Python the dictionary format
                        while file is None:
                            print("Failed reading fitted param")
                            file = open("../../parameters/simulations/fitted"+f"_{init_param.index(param)}_{randomInstance}.yaml","r")
                        
                        universe_params = yaml.load(file, Loader=yaml.FullLoader)
                        
                        while universe_params is None:
                            print("Failed converting to yaml fitted param")
                            universe_params = yaml.load(file, Loader=yaml.FullLoader)

                    # Read initialization
                    with open(f"../../parameters/simulations/oct21_{init_param.index(param)}_{randomInstance}.yaml","r") as file:
                        # The FullLoader parameter handles the conversion from YAML
                        # scalar values to Python the dictionary format
                        while file is None:
                            print("Failed reading init param")
                            file = open(f"../../parameters/simulations/oct21_{init_param.index(param)}_{randomInstance}.yaml","r")

                        initialization = yaml.load(file, Loader=yaml.FullLoader)
                        while initialization is None:
                            print("Failed converting to yaml init param")
                            initialization = yaml.load(file, Loader=yaml.FullLoader)
                    
                # Read econ parameters
                    with open(f"../../parameters/simulations/econ_{init_param.index(param)}_{randomInstance}.yaml","r") as file:
                        while file is None:
                            print("Failed reading econ param")
                            file = open(f"../../parameters/simulations/econ_{init_param.index(param)}_{randomInstance}.yaml","r")

                        econ_params = yaml.load(file, Loader=yaml.FullLoader)
                        while econ_params is None:
                            print("Failed converting to yaml econ param")
                            econ_params = yaml.load(file, Loader=yaml.FullLoader)




                    start_day = result["experiment_params"]["start_day"]


                    experiment_params = result["experiment_params"]

                    dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, 1, experiment_params["T"], universe_params["mixing"], start_day, result["experiment_params"]['eta'], extra_data = True)

                    for t in range(experiment_params["T"]):
                        dynModel.take_time_step(result["m_tests"][t], result["a_tests"][t], result["policy"][t])

                    dynModel.take_end_steps()

                    deaths = dynModel.get_total_deaths()
                    econ_value = (dynModel.get_total_reward() + dynModel.get_total_deaths() * xi) / 1e9
                    reward = dynModel.get_total_reward()/1e9
                    economic_losses = V/1e9 - econ_value
                    total_loss = V/1e9 - reward


                    
                    data = {
                            "init_u_hat":u_hat,
                            "xi":round(result["experiment_params"]["xi"] / 37199.03),
                            "icus":result["experiment_params"]["icus"],
                            "n_m_tests":result["experiment_params"]["n_m_tests"],
                            "eta":result["experiment_params"]["eta"],
                            "init_param":param,
                            "randomInstance":randomInstance,
                            "economics_value":econ_value,
                            "deaths":deaths,
                            "reward":reward,
                            "economic_losses":economic_losses,
                            "total_loss":total_loss
                        }
                    
                    new_data.append(data)
                    # print(new_data)
                    new_data_df = pd.DataFrame(new_data)
                    # print(new_data_df)
                    # print(old_data)
                    old_data = old_data.append(new_data_df, ignore_index=True)
                    old_data.to_excel(f"plots/results-Robustness.xlsx", index=False)
                    # print(old_data)

                    new_data = []
                    
                    print(instance)
                else:
                    print("Found in database")
                    print(instance)


                               
        print(h)

        # # print(new_data)
        # new_data_df = pd.DataFrame(new_data)
        # # print(new_data_df)
        # # print(old_data)
        # old_data.append(new_data_df, ignore_index=True)
        # old_data.to_excel(f"plots/results-Robustness.xlsx", index=False)
        # # print(old_data)

        new_data = []