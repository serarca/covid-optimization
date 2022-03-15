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

etas = [0.1]
deltas = [0.5]
icus = [2900]
lockdown_freq=14
testing_freq=7
nmtests = [0]
natests = [0]


# Modify rewards to losses

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

file_name = f"plots/results-Fig2.xlsx"

try:
   df = pd.read_excel(file_name, engine='openpyxl')
except FileNotFoundError:
    print("File Not Found, check file name")

df["economic_losses"] = V/1e9 - df["economics_value"]
df["total_loss"] = (V - df["reward"])/1e9

df.to_excel(file_name, index=False)


# heurs = ["linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient",
# "tree_heuristic_featureSet=NON_TARGETED_treeDepth=4",
# "tree_heuristic_featureSet=ALL_2_treeDepth=4",
# "tree_heuristic_featureSet=TARGETED_AGE_treeDepth=4",
# "tree_heuristic_featureSet=TARGETED_CONT_GIVEN_treeDepth=4",
# "tree_heuristic_featureSet=TARGETED_ECON_GRAD_NORM_treeDepth=4",
# "tree_heuristic_featureSet=TARGETED_P_D_treeDepth=4",
# "tree_heuristic_featureSet=TARGETED_R_PERC_treeDepth=4",
# "tree_heuristic_featureSet=TARGETED_VG_LIFE_treeDepth=4",
# "tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=4",
# "tree_heuristic_featureSet=ALL_2_treeDepth=10"
# ]

# "tree_heuristic_featureSet=NON_TARGETED_treeDepth=6",
# "tree_heuristic_featureSet=ALL_2_treeDepth=6",
# "tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=6",
# "tree_heuristic_featureSet=NON_TARGETED_treeDepth=7",
# "tree_heuristic_featureSet=ALL_2_treeDepth=7",
# "tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=7",

# "tree_heuristic_featureSet=SIMPLE_PLUS_treeDepth=4", "tree_heuristic_featureSet=SIMPLE_treeDepth=4",
# "tree_heuristic_featureSet=ORIGINAL_PLUS_treeDepth=4",
# "tree_heuristic_featureSet=ORIGINAL_treeDepth=4",
# "tree_heuristic_featureSet=ENGINEERED_1_MINUS_treeDepth=4","tree_heuristic_featureSet=ENGINEERED_1_PLUS_treeDepth=4","French_trigger", "ICU_admissions_trigger",


#  "French_trigger", "ICU_admissions_trigger"
#"full_open","full_lockdown",
# ["linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True", "linearization_heuristic_optBouncing=False_initial_uhat=age_group_gradient_targetGroups=True_targetAct=False_targetTests=True", "linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient","linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True", "full_lockdown", "full_open"]
# "linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True"
# linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True
# ,"dynamic_gradient"]
#  ["linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient"]
#  ["dynamic_gradient"]
# ["linearization_heuristic_optBouncing=False_initial_uhat=full_open"]
# ,"dynamic_gradient","linearization_heuristic_optBouncing=False_initial_uhat=full_lockdown","real","constant_gradient"]
# ["dynamic_gradient"]
# , "linearization_heuristic_optBouncing=False_initial_uhat=full_lockdown"]

# xi-0.0_icus-2000.0_testing-linearization_heuristic_natests-0.0_nmtests-0.0_T-90_startday-60_groups-all_dschool-0.5_eta-0_lockdownFreq-14_testingFreq-7

fig1, ax1 = plt.subplots(1)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis



fig2, ax2 = plt.subplots(1)
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])




markers = [".","D","s","*","+","p","x"]
marker = 0

heuristic_labels={}
marker_dict = {}

heuristic_labels["linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient"] = "ROLD AGE-ACT"
heuristic_labels["French_trigger"] = "Hybrid-t AND"
heuristic_labels["French_trigger_or"] = "Hybrid-t OR"
heuristic_labels["ICU_admissions_trigger"] = "ICU-t"
heuristic_labels["linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True"] = "ROLD ACT"

heuristic_labels["tree_heuristic_featureSet=SIMPLE_PLUS_treeDepth=4"] = "Tree Heur - SIMPLE PLUS - D 4"

heuristic_labels["tree_heuristic_featureSet=SIMPLE_treeDepth=4"] = "Tree Heur - SIMPLE - D 4"

heuristic_labels["tree_heuristic_featureSet=ORIGINAL_PLUS_treeDepth=4"] = "Tree Heur - ORIGINAL PLUS - D 4"

heuristic_labels["tree_heuristic_featureSet=ORIGINAL_treeDepth=4"] = "Tree Heur - ORIGINAL - D 4"

heuristic_labels["tree_heuristic_featureSet=ENGINEERED_1_PLUS_treeDepth=4"] = "Tree Heur - ENGINEERED 1 PLUS - D 4"

heuristic_labels["tree_heuristic_featureSet=ENGINEERED_1_MINUS_treeDepth=4"] = "Tree Heur - ENGINEERED 1 MINUS - D 4"

heuristic_labels["tree_heuristic_featureSet=NON_TARGETED_treeDepth=4"] = "Tree Heur - NON TARGETED - D 4"

heuristic_labels["tree_heuristic_featureSet=ALL_2_treeDepth=4"] = "Tree Heur - ALL 2 - D 4"

heuristic_labels["tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=4"] = "Tree Heur - TARGETED ECON CONT RATIO - D 4"

heuristic_labels["tree_heuristic_featureSet=NON_TARGETED_treeDepth=6"] = "Tree Heur - NON TARGETED - D 6"

heuristic_labels["tree_heuristic_featureSet=ALL_2_treeDepth=6"] = "Tree Heur - ALL 2 - D 6"

heuristic_labels["tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=6"] = "Tree Heur - TARGETED ECON CONT RATIO - D 6"

heuristic_labels["tree_heuristic_featureSet=NON_TARGETED_treeDepth=10"] = "Tree Heur - NON TARGETED - D 10"

heuristic_labels["tree_heuristic_featureSet=ALL_2_treeDepth=10"] = "Tree Heur - ALL 2 - D 7"

heuristic_labels["tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=10"] = "Tree Heur - TARGETED ECON CONT RATIO - D 10"

heuristic_labels["tree_heuristic_featureSet=TARGETED_AGE_treeDepth=4"]= "Tree Heur - TARGETED_AGE - D 4"
heuristic_labels["tree_heuristic_featureSet=TARGETED_CONT_GIVEN_treeDepth=4"]= "Tree Heur - TARGETED_CONT_GIVEN - D 4"
heuristic_labels["tree_heuristic_featureSet=TARGETED_ECON_GRAD_NORM_treeDepth=4"]= "Tree Heur - TARGETED_ECON_GRAD_NORM - D 4"
heuristic_labels["tree_heuristic_featureSet=TARGETED_P_D_treeDepth=4"]= "Tree Heur - TARGETED_P_D - D 4"
heuristic_labels["tree_heuristic_featureSet=TARGETED_R_PERC_treeDepth=4"]= "Tree Heur - TARGETED_R_PERC - D 4"
heuristic_labels["tree_heuristic_featureSet=TARGETED_VG_LIFE_treeDepth=4"]= "Tree Heur - TARGETED_VG_LIFE - D 4"




heuristic_labels["linearization_heuristic_optBouncing=False_initial_uhat=age_group_gradient_targetGroups=True_targetAct=False_targetTests=True"] = "ROLD AGE"

heuristic_labels["linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True"] = "ROLD NO-TARGET"

heuristic_labels["full_lockdown"] = "Full Confinement"
heuristic_labels["full_open"] = "Full Open"

marker_dict["linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient"] = "*"
marker_dict["French_trigger"] = "+"
marker_dict["French_trigger_or"] = "."
marker_dict["ICU_admissions_trigger"] = "h"
marker_dict["linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True"] = "D"
marker_dict["tree_heuristic_featureSet=SIMPLE_PLUS_treeDepth=4"] = "h"

marker_dict["tree_heuristic_featureSet=SIMPLE_treeDepth=4"] = "+"

marker_dict["tree_heuristic_featureSet=ORIGINAL_PLUS_treeDepth=4"] = "."

marker_dict["tree_heuristic_featureSet=ORIGINAL_treeDepth=4"] = "x"

marker_dict["tree_heuristic_featureSet=ENGINEERED_1_PLUS_treeDepth=4"] = "d"

marker_dict["tree_heuristic_featureSet=ENGINEERED_1_MINUS_treeDepth=4"] = "X"

marker_dict["tree_heuristic_featureSet=NON_TARGETED_treeDepth=4"] = "h"

marker_dict["tree_heuristic_featureSet=ALL_2_treeDepth=4"] = "."

marker_dict["tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=4"] = "x"
marker_dict["tree_heuristic_featureSet=NON_TARGETED_treeDepth=6"] = "X"

marker_dict["tree_heuristic_featureSet=ALL_2_treeDepth=6"] = "d"

marker_dict["tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=6"] = "o"

marker_dict["tree_heuristic_featureSet=NON_TARGETED_treeDepth=10"] = "+"

marker_dict["tree_heuristic_featureSet=ALL_2_treeDepth=10"] = "D"

marker_dict["tree_heuristic_featureSet=TARGETED_ECON_CONT_RATIO_treeDepth=10"] = "+"

marker_dict["tree_heuristic_featureSet=TARGETED_AGE_treeDepth=4"]= "X"
marker_dict["tree_heuristic_featureSet=TARGETED_CONT_GIVEN_treeDepth=4"]= "X"
marker_dict["tree_heuristic_featureSet=TARGETED_ECON_GRAD_NORM_treeDepth=4"]= "X"
marker_dict["tree_heuristic_featureSet=TARGETED_P_D_treeDepth=4"]= "X"
marker_dict["tree_heuristic_featureSet=TARGETED_R_PERC_treeDepth=4"]= "X"
marker_dict["tree_heuristic_featureSet=TARGETED_VG_LIFE_treeDepth=4"]= "X"


marker_dict["full_lockdown"] = r"$FC$"
marker_dict["full_open"] = r"$FO$"

handles_ax1 = []
handles_ax2 = []

# try:
#     old_data = pd.read_excel("plots/results-Fig1.xlsx", engine='openpyxl')
# except FileNotFoundError:
#     old_data = pd.DataFrame([{
#                                             "lock_heuristic":"",
#                                             "xi":0,
#                                             "icus":0,
#                                             "n_m_tests":0,
#                                             "eta":0,
#                                             "economics_value":0,
#                                             "deaths":0,
#                                             "reward":0,
#                                         }])

# # new_data = []


# for h in heurs:
#     for e in etas:
#         for d in deltas:
#             for i in icus:
#                 for nm in nmtests:
#                     for na in natests:
#                         all_deaths = []
#                         all_econ_values = []
#                         all_xis = []
#                         all_rewards = []

                        
#                         symbols = ["*",".","d","o","s","^"]
                        
#                         for file_name in os.listdir(h):
#                             instance = file_name.split("_")

#                             for p in instance:
#                                 if "xi" in p:
#                                     xi = float(p.split("-")[1])
#                     #                 print(xi)
#                                 if "icus" in p:
#                                     icu = float(p.split("-")[1])
#                     #                 print(icu)
#                                 if "natests" in p:
#                                     natest = float(p.split("-")[1])
#                     #                 print(natest)
#                                 if "nmtests" in p:
#                                     nmtest = float(p.split("-")[1])
#                     #                 print(nmtest)
#                                 if "dschool" in p:
#                                     delta = float(p.split("-")[1])
#                     #                 print(delta)
#                                 if "eta" in p:
#                                     eta = float(p.split("-")[1])
#                     #                 print(eta)
#                                 if "T" in p:
#                                     T = float(p.split("-")[1])
#                     #                 print(eta)

#                             if eta == e and icu == i and nmtest == nm and natest == na and delta == d and T==90:
#                                 # print(h)
#                                 # print(instance)
                                
#                                 # print(old_data.loc[(old_data["lock_heuristic"] == h) & ((old_data["xi"] <= ((xi/37199.03) + 1)) & (old_data["xi"] >= ((xi/37199.03) - 1))) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["deaths"])

#                                 if not old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)].empty:

                                    
#                                     # print("Found in results")
#                                     deaths = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["deaths"].item()
#                                     econ_value = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["economic_losses"].item()
#                                     reward = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["total_loss"].item()
#                                     # print(f"reward is {reward}")
#                                     # print(f"Econ value rerun: {econ_value}")
#                                     # with open("%s/%s"%(h,file_name)) as file:
#                                     #     result = yaml.load(file, Loader=yaml.UnsafeLoader)
#                                     # print(f"Econ value original: {result['results']['economics_value']/1e9}")
                                    
#                                 else:


#                                     with open("%s/%s"%(h,file_name)) as file:
#                                         result = yaml.load(file, Loader=yaml.UnsafeLoader)


#                                     # Read group parameters
#                                     with open("../../parameters/fitted.yaml") as file:
#                                         universe_params_all = yaml.load(file, Loader=yaml.FullLoader)

#                                     # Read econ parameters
#                                     with open("../../parameters/econ.yaml") as file:
#                                         econ_params_all = yaml.load(file, Loader=yaml.FullLoader)

#                                     # Read group parameters
#                                     with open("../../parameters/one_group_fitted.yaml") as file:
#                                         universe_params_one = yaml.load(file, Loader=yaml.FullLoader)

#                                     # Read econ parameters
#                                     with open("../../parameters/one_group_econ.yaml") as file:
#                                         econ_params_one = yaml.load(file, Loader=yaml.FullLoader)


                                    

#                                     start_day = 0

#                                     universe_params = universe_params_all
#                                     econ_params = econ_params_all

#                                     with open("../../initialization/oct21.yaml") as file:
#                                         initialization = yaml.load(file, Loader=yaml.FullLoader)
                                    

#                                     experiment_params = result["experiment_params"]

#                                     dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, 1, experiment_params["T"], universe_params["mixing"], start_day, result["experiment_params"]['eta'], extra_data = True)

#                                     for t in range (experiment_params["T"]):
#                                         for ag in result['policy'][t]:
                                            
#                                             if 'home' not in result['policy'][t][ag]:
#                                                 result['policy'][t][ag]['home'] = 1.0
                                        
#                                         dynModel.take_time_step(result["m_tests"][t], result["a_tests"][t], result["policy"][t])

#                                     dynModel.take_end_steps()

#                                     deaths = dynModel.get_total_deaths()
#                                     econ_value = (dynModel.get_total_reward() + dynModel.get_total_deaths() * xi) / 1e9
#                                     reward = dynModel.get_total_reward()
                                    
#                                     data = {
#                                             "lock_heuristic":h,
#                                             "xi":round(result["experiment_params"]["xi"] / 37199.03),
#                                             "icus":result["experiment_params"]["icus"],
#                                             "n_m_tests":result["experiment_params"]["n_m_tests"],
#                                             "eta":result["experiment_params"]["eta"],
#                                             "economics_value":float((dynModel.get_total_reward() + dynModel.get_total_deaths() * xi) / 1e9),
#                                             "deaths":float(dynModel.get_total_deaths()),
#                                             "reward":float(dynModel.get_total_reward()),
#                                         }
                                    
#                                     new_data.append(data)






#                                 all_deaths.append(deaths)
#                                 all_econ_values.append(econ_value)
#                                 all_xis.append(round(xi / 37199.03))
#                                 all_rewards.append(reward)
                        
                        
                        
                        
                        
#                         all_rewards = [x for _, x in sorted(zip(all_xis, all_rewards))]
#                         all_deaths = [x for _, x in sorted(zip(all_xis, all_deaths))]
#                         all_econ_values = [x for _, x in sorted(zip(all_xis, all_econ_values))]
#                         all_xis = sorted(all_xis)
                        
#                         print(f"Number of diff points: {max(len(np.unique(np.array(all_deaths))),len(np.unique(np.array(all_econ_values))))}")
#                         print(f"Number of xis: {len(all_xis)}")
#                         print(all_xis)

#                         max_xis = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 800, 810, 820, 840, 850, 860, 870, 900, 910, 930, 940, 950, 960, 970, 980, 990, 10000000000000000, 510, 11, 880, 790, 920, 890, 380, 830}
#                         print(h)
#                         print(max_xis.difference(set(all_xis)))



#                         # print(all_rewards)
                        
                        
#                         restricted_xi = [x for x in all_xis if x < 1000]
#                         restricted_reward = all_rewards[:len(restricted_xi)]

#                         restricted_deaths = all_deaths[:len(restricted_xi)]

#                         restricted_econ_values = all_econ_values[:len(restricted_xi)]

#                         if h != "full_lockdown" and h!= "full_open":
#                             handles_ax1 += [ax1.scatter(restricted_xi, restricted_reward, label=heuristic_labels[h],alpha=0.6, marker=marker_dict[h])]

#                             handles_ax2 += [ax2.scatter(restricted_deaths, restricted_econ_values,label=heuristic_labels[h],alpha=0.5, marker=marker_dict[h])]
#                         # , color=["r","g","b", "c","m","y"])]

#                         if h == "full_lockdown": 
                        
#                             handles_ax2 += [ax2.scatter(all_deaths, all_econ_values,label=heuristic_labels[h],alpha=0.9,s=90, color="C6", marker=marker_dict[h], edgecolors="none")]
#                         if h == "full_open":
#                             handles_ax2 += [ax2.scatter(all_deaths, all_econ_values,label=heuristic_labels[h],alpha=0.9,s=90, color="C7", marker=marker_dict[h], edgecolors="none")]
#                         marker += 1

#                         # print(all_econ_values)
#                         # print(all_deaths)

# print(new_data)
# new_data_df = pd.DataFrame(new_data)
# print(new_data_df)
# print(old_data)
# old_data.append(new_data_df, ignore_index=True).to_excel(f"plots/results-Fig1.xlsx", index=False)
# print(old_data)


        
# # for eta in etas:
# #     for delta in deltas:
# #         for icu in icus:
# #             for nmtest in nmtests:
# #                 for natest in natests:
# #                     for heur in heurs:
    
# # for x,y in zip(all_deaths, all_econ_values):

# #     label = "{:.2f}".format(all_xis[all_deaths.index(x)])

#     # ax2.annotate(label, # this is the text
#     #                     (x,y), # this is the point to label
#     #                     textcoords="offset points", # how to position the text
#     #                     xytext=(0,10), # distance from text to points (x,y)
#     #                     ha='center') # horizontal alignment can be left, right or center





# # Modify rewards to losses

# T = 104

# experiment_params = {
#   "T":90,
#     "delta_schooling":0.5,
#     "eta":0.1,
#     "icus":2900.0,
#     "lockdown_freq":14,
#     "n_a_tests":0,
#     "n_m_tests":60000.0,
#     "start_day":0,
#     "test_freq":7,
#     'xi':0.0
#     }

# # Read group parameters
# with open("../../parameters/fitted.yaml") as file:
#     universe_params_all = yaml.load(file, Loader=yaml.FullLoader)

# # Read econ parameters
# with open("../../parameters/econ.yaml") as file:
#     econ_params_all = yaml.load(file, Loader=yaml.FullLoader)

# start_day = experiment_params["start_day"]


# universe_params = universe_params_all
# econ_params = econ_params_all

# with open("../../initialization/oct21.yaml") as file:
#     initialization = yaml.load(file, Loader=yaml.FullLoader)

# experiment_params = experiment_params

# dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, 1, experiment_params["T"], universe_params["mixing"], start_day, experiment_params['eta'], extra_data = True)



# econ_activities = ["transport","leisure","other"]

# V = 0

# for age_group in dynModel.age_groups:
#     total_initial_population_g = sum([dynModel.initialization[age_group][cat] for cat in ["N","E","I","ICU","Ia","Ims","Ips","Iss","Rq"] ])

# #   D: 2.0129755906530393
# #   E: 2285.276888763268
# #   H: 13.540885275895503
# #   I: 2142.166336075084
# #   ICU: 3.5669812156992835
# #   Ia: 0.0
# #   Ims: 0.0
# #   Ips: 0.0
# #   Iss: 0.0
# #   N: 1594534.826129878
# #   R: 160265.76119148443
# #   Rq: 302.0529959235804
# #   S: 1429841.621713554
#     # print(f"age group: {age_group}")
#     # print(f"Initial Pop: {total_initial_population_g}")
#     v_g = sum([dynModel.econ_params["employment_params"]["v"][age_group][activity] for activity in econ_activities])
    
#     v_employment_1_g = v_g

#     # print(f"employment: {v_g * 365}")

#     v_schooling_1_g =  (

#         dynModel.experiment_params['delta_schooling']*dynModel.econ_params['schooling_params'][age_group]
#     )


#     V += (v_employment_1_g + v_schooling_1_g) * T * total_initial_population_g

# print(V)

# file_name = f"plots/results-Fig1.xlsx"

# try:
#    df = pd.read_excel(file_name, engine='openpyxl')
# except FileNotFoundError:
#     print("File Not Found, check file name")

# df["economic_losses"] = V/1e9 - df["economics_value"]
# df["total_loss"] = (V - df["reward"])/1e9

# df.to_excel(file_name, index=False)


# ax1.set(ylabel='Total Losses (Billion €)')
# ax1.set(xlabel=r'Emotional Cost of Death $\chi$ ($\times$ annual GDP/capita)')
# ax1.xaxis.set_major_locator(plt.MaxNLocator(3))

# # Put a legend to the right of the current axis
# # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# # ax1.set_title(f"xi vs totalObj eta={etas[0]} icu={icus[0]}")
# fig1.savefig(f"plots/Fig1_xi_vs_totalObj_eta={etas[0]}_icu={icus[0]}_treesHeur.pdf", bbox_inches='tight')
# # fig1.legend(loc='center left', bbox_to_anchor=(1, 0.5))



# ax2.set(ylabel='Economic Losses (Billion €)')
# ax2.set(xlabel='Total Number of Deaths')


# # fig2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# # ax2.set_title(f"deaths vs econ eta={etas[0]} icu={icus[0]}")
# fig2.savefig(f"plots/Fig1_deaths_vs_econLosses_eta={etas[0]}_icu={icus[0]}_treesHeur.pdf", bbox_inches='tight')



etas = [0.1]
deltas = [0.5]
icus = [2900]
lockdown_freq=14
testing_freq=7
nmtests = [0]
natests = [0]

heurs = ["linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True", "linearization_heuristic_optBouncing=False_initial_uhat=age_group_gradient_targetGroups=True_targetAct=False_targetTests=True", "linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True"]

# ,  "linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient"]

#  "linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True", "linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient"
# ,"linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient","linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True","linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True", "linearization_heuristic_optBouncing=False_initial_uhat=age_group_gradient_targetGroups=True_targetAct=False_targetTests=True" 
# ["linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True", "linearization_heuristic_optBouncing=False_initial_uhat=age_group_gradient_targetGroups=True_targetAct=False_targetTests=True", "linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient","linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True", "full_lockdown", "full_open"]
# "linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True"
# linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True
# ,"dynamic_gradient"]
#  ["linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient"]
#  ["dynamic_gradient"]
# ["linearization_heuristic_optBouncing=False_initial_uhat=full_open"]
# ,"dynamic_gradient","linearization_heuristic_optBouncing=False_initial_uhat=full_lockdown","real","constant_gradient"]
# ["dynamic_gradient"]
# , "linearization_heuristic_optBouncing=False_initial_uhat=full_lockdown"]

# xi-0.0_icus-2000.0_testing-linearization_heuristic_natests-0.0_nmtests-0.0_T-90_startday-60_groups-all_dschool-0.5_eta-0_lockdownFreq-14_testingFreq-7

fig1, ax1 = plt.subplots(1)
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis



fig2, ax2 = plt.subplots(1)
box = ax2.get_position()
ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

ax2.set_ylim([0, 14])


markers = [".","D","s","*","+","p","x"]
marker = 0

heuristic_labels={}
marker_dict = {}

heuristic_labels["linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient"] = "Age and Activities"
heuristic_labels["French_trigger"] = "French Trigger Heuristic (and)"
heuristic_labels["French_trigger_or"] = "French Trigger Heuristic (or)"
heuristic_labels["ICU_admissions_trigger"] = "ICU Admission Trigger Heuristic"
heuristic_labels["linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True"] = "Activities only"

heuristic_labels["linearization_heuristic_optBouncing=False_initial_uhat=age_group_gradient_targetGroups=True_targetAct=False_targetTests=True"] = "Age only"

heuristic_labels["linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True"] = "No Targeting"

heuristic_labels["full_lockdown"] = "Full Confinement"
heuristic_labels["full_open"] = "Full Open"

heuristic_labels["French_trigger"] = "Hybrid-t AND"
heuristic_labels["French_trigger_or"] = "Hybrid-t OR"
heuristic_labels["ICU_admissions_trigger"] = "ICU-t"



marker_dict["linearization_heuristic_optBouncing=False_initial_uhat=dynamic_gradient"] = "*"
marker_dict["French_trigger"] = "+"
marker_dict["French_trigger_or"] = "."
marker_dict["ICU_admissions_trigger"] = "h"
marker_dict["linearization_heuristic_optBouncing=False_initial_uhat=activity_gradient_targetGroups=False_targetAct=True_targetTests=True"] = "X"

marker_dict["linearization_heuristic_optBouncing=False_initial_uhat=age_group_gradient_targetGroups=True_targetAct=False_targetTests=True"] = "P"

marker_dict["linearization_heuristic_optBouncing=False_initial_uhat=time_gradient_targetGroups=False_targetAct=False_targetTests=True"] = "D"


marker_dict["full_lockdown"] = r"$FC$"
marker_dict["full_open"] = r"$FO$"

handles_ax1 = []
handles_ax2 = []

try:
    old_data = pd.read_excel("plots/results-Fig2.xlsx",engine='openpyxl')
except FileNotFoundError:
    old_data = pd.DataFrame([{
                                            "lock_heuristic":"",
                                            "xi":0,
                                            "icus":0,
                                            "n_m_tests":0,
                                            "eta":0,
                                            "economics_value":0,
                                            "deaths":0,
                                            "reward":0,
                                        }])

new_data = []


for h in heurs:
    for e in etas:
        for d in deltas:
            for i in icus:
                for nm in nmtests:
                    for na in natests:
                        all_deaths = []
                        all_econ_values = []
                        all_xis = []
                        all_rewards = []

                        
                        symbols = ["*",".","d","o","s","^"]
                        
                        for file_name in os.listdir(h):
                            instance = file_name.split("_")

                            for p in instance:
                                if "xi" in p:
                                    xi = float(p.split("-")[1])
                    #                 print(xi)
                                if "icus" in p:
                                    icu = float(p.split("-")[1])
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
                                if "eta" in p:
                                    eta = float(p.split("-")[1])
                    #                 print(eta)
                                if "T" in p:
                                    T = float(p.split("-")[1])
                    #                 print(eta)

                            if eta == e and icu == i and nmtest == nm and natest == na and delta == d and T==90:
                                # print(h)
                                # print(instance)
                                
                                # print(old_data.loc[(old_data["lock_heuristic"] == h) & ((old_data["xi"] <= ((xi/37199.03) + 1)) & (old_data["xi"] >= ((xi/37199.03) - 1))) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["deaths"])

                                if not old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)].empty:

                                    
                                    # print("Found in results")
                                    deaths = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["deaths"].item()
                                    econ_value = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["economic_losses"].item()
                                    reward = old_data.loc[(old_data["lock_heuristic"] == h) & (old_data["xi"] == round(xi/37199.03)) & (old_data["icus"] == i) & (old_data["n_m_tests"] == nm) & (old_data["eta"] == e)]["total_loss"].item()
                                    # print(f"reward is {reward}")
                                    
                                    # print(f"Econ value rerun: {econ_value}")
                                    # with open("%s/%s"%(h,file_name)) as file:
                                    #     result = yaml.load(file, Loader=yaml.UnsafeLoader)
                                    # printf(f"Econ value original: {result['results']['econ_value']/10e9}")

                                else:


                                    with open("%s/%s"%(h,file_name)) as file:
                                        result = yaml.load(file, Loader=yaml.UnsafeLoader)


                                    # Read group parameters
                                    with open("../../parameters/fitted.yaml") as file:
                                        universe_params_all = yaml.load(file, Loader=yaml.FullLoader)

                                    # Read econ parameters
                                    with open("../../parameters/econ.yaml") as file:
                                        econ_params_all = yaml.load(file, Loader=yaml.FullLoader)

                                    # Read group parameters
                                    with open("../../parameters/one_group_fitted.yaml") as file:
                                        universe_params_one = yaml.load(file, Loader=yaml.FullLoader)

                                    # Read econ parameters
                                    with open("../../parameters/one_group_econ.yaml") as file:
                                        econ_params_one = yaml.load(file, Loader=yaml.FullLoader)


                                    start_day = result["experiment_params"]["start_day"]

                                    if result["groups"] == "one":
                                        universe_params = universe_params_one
                                        econ_params = econ_params_one

                                        with open("../../initialization/%ddays_one_group.yaml"%start_day) as file:
                                            initialization = yaml.load(file, Loader=yaml.FullLoader)

                                    elif result["groups"] == "all":
                                        universe_params = universe_params_all
                                        econ_params = econ_params_all

                                        with open("../../initialization/oct21.yaml") as file:
                                            initialization = yaml.load(file, Loader=yaml.FullLoader)
                                    else:
                                        assert(False)

                                    experiment_params = result["experiment_params"]

                                    dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, 1, experiment_params["T"], universe_params["mixing"], start_day, result["experiment_params"]['eta'], extra_data = True)

                                    for t in range(experiment_params["T"]):
                                        dynModel.take_time_step(result["m_tests"][t], result["a_tests"][t], result["policy"][t])

                                    dynModel.take_end_steps()

                                    deaths = dynModel.get_total_deaths()
                                    econ_value = (dynModel.get_total_reward() + dynModel.get_total_deaths() * xi) / 10e9
                                    reward = dynModel.get_total_reward()
                                    
                                    data = {
                                            "lock_heuristic":h,
                                            "xi":round(result["experiment_params"]["xi"] / 37199.03),
                                            "icus":result["experiment_params"]["icus"],
                                            "n_m_tests":result["experiment_params"]["n_m_tests"],
                                            "eta":result["experiment_params"]["eta"],
                                            "economics_value":(dynModel.get_total_reward() + dynModel.get_total_deaths() * xi) / 1e9,
                                            "deaths":dynModel.get_total_deaths(),
                                            "reward":dynModel.get_total_reward() /1e9,
                                        }
                                    
                                    new_data.append(data)






                                all_deaths.append(deaths)
                                all_econ_values.append(econ_value)
                                all_xis.append(round(xi / 37199.03))
                                all_rewards.append(reward)

                                
                        
                        
                        
                        
                        
                        all_rewards = [x for _, x in sorted(zip(all_xis, all_rewards))]
                        all_deaths = [x for _, x in sorted(zip(all_xis, all_deaths))]
                        all_econ_values = [x for _, x in sorted(zip(all_xis, all_econ_values))]
                        all_xis = sorted(all_xis)
                        print(f"Number of diff points: {max(len(np.unique(np.array(all_deaths))),len(np.unique(np.array(all_econ_values))))}")
                        print(f"Number of xis: {len(all_xis)}")
                        print(all_xis)

                        max_xis = {0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710, 720, 730, 740, 750, 760, 770, 780, 800, 810, 820, 840, 850, 860, 870, 900, 910, 930, 940, 950, 960, 970, 980, 990, 10000000000000000, 510, 11, 880, 790, 920, 890, 380, 830}

                        
                        print(h)
                        print(max_xis.difference(set(all_xis)))

                        restricted_xi = [x for x in all_xis if x < 2000]
                        restricted_reward = all_rewards[:len(restricted_xi)]
                        restricted_deaths = all_deaths[:len(restricted_xi)]
                        restricted_econ_values = all_econ_values[:len(restricted_xi)]

                        print(f"length of restricted xi: {len(restricted_xi)}")



                        if h != "full_lockdown" and h!= "full_open":
                            handles_ax1 += [ax1.scatter(restricted_xi, restricted_reward, label=heuristic_labels[h],alpha=0.5, marker=marker_dict[h])]

                            handles_ax2 += [ax2.scatter(restricted_deaths, restricted_econ_values,label=heuristic_labels[h],alpha=0.5, marker=marker_dict[h])]
                        # , color=["r","g","b", "c","m","y"])]

                        if h == "full_lockdown": 
                        
                            handles_ax2 += [ax2.scatter(all_deaths, all_econ_values,label=heuristic_labels[h],alpha=0.9,s=90, color="C6", marker=marker_dict[h], edgecolors="none")]
                        if h == "full_open":
                            handles_ax2 += [ax2.scatter(all_deaths, all_econ_values,label=heuristic_labels[h],alpha=0.9,s=90, color="b", marker=marker_dict[h], edgecolors="none")]
                    
                        marker += 1

                        # print(all_econ_values)
                        # print(all_deaths)

# print(new_data)
new_data_df = pd.DataFrame(new_data)
# print(new_data_df)
# print(old_data)
old_data.append(new_data_df, ignore_index=True).to_excel(f"plots/results-Fig2.xlsx", index=False)
# print(old_data)



# ax1.set(ylabel='Total Losses (Billion €)')
# ax1.set(xlabel=r'Emotional Cost of Death $\chi$ ($\times$ annual GDP/capita)')
# ax1.xaxis.set_major_locator(plt.MaxNLocator(3))

# # Put a legend to the right of the current axis
# # ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# ax1.legend(loc='best')
# # , handles=handles_ax1)

# # ax1.set_title(f"xi vs totalObj eta={etas[0]} icu={icus[0]}")
# fig1.savefig(f"plots/presentation-Fig2_xi_vs_totalObj_eta={etas[0]}_icu={icus[0]}.pdf", bbox_inches='tight')



ax2.set(ylabel='Economic Losses (Billion €)')
ax2.set(xlabel='Total Number of Deaths')


# fig2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

ax2.legend(loc='best')
# , handles=handles_ax2)

# ax2.set_title(f"deaths vs econ eta={etas[0]} icu={icus[0]}")
fig2.savefig(f"plots/z3presentation-Fig2_deaths_vs_econLosses_eta={etas[0]}_icu={icus[0]}.pdf", bbox_inches='tight')
     


