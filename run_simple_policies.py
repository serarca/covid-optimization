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
from itertools import chain

#from joblib import Parallel, delayed


def main():

    instance_index = 0

    if len(sys.argv) > 1:
        instance_index = int(sys.argv[1])

    # 30 * 37199.03
    # Some paramters to test the linearization heuristic
    xi_mult_values = [1, 3, 4, 6, 7, 8, 9]
    # chain(range(0,200,5), range(0, 1000, 10))
    testing_values = [0]
    # [0, 30000, 60000, 120000]
    icu_values = [2900]
    # [2000, 2300, 2600, 2900, 3200]

    params_to_try = {
        "delta_schooling":[0.5],
        "xi":[mult * 37199.03 for mult in xi_mult_values],
        "icus":[ic for ic in icu_values],
        "mtests":[test_cap for test_cap in testing_values],
        # "atests":[test_cap / scaling for test_cap in testing_values],
        "frequencies":[(7,14)],
        "region":["fitted-scaled"], 
        "econ": ["econ-scaled"],
        "init": ["oct21-scaled"],
        "eta":[0.1],
        "trust_region_radius":[0.05],
        "max_inner_iterations_mult":[2],
        "trigger_policy":["French_trigger_or_benchmark_ref","French_trigger_benchmark_ref","ICU_admissions_trigger_benchmark_ref"]
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
    trigger_policy = all_instances[instance_index][11]
    
    path = os.getcwd()
    print(path)
    os.chdir(f"{path}/benchmarks")

    os.system(f"python3 {trigger_policy}.py --delta {delta} --icus {icus} --eta {eta} --groups {groups} --xi {xi} --a_tests {atests} --m_tests {mtests}")

    os.chdir(path)
 


if __name__ == "__main__":
    main()
