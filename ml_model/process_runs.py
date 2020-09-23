import yaml
from inspect import getsourcefile
import os.path
import os

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

from group import SEIR_group, DynamicalModel
import math
import pprint
import pandas as pd
import pickle
import numpy as np
from fast_group import FastDynamicalModel
from aux import *
from scipy.optimize import Bounds,minimize,LinearConstraint
from copy import deepcopy


results = []
print("Files to process: ",len(os.listdir("runs/")))
for i,n in enumerate(os.listdir("runs/")):
	if "run" in n:
		print(i/len(os.listdir("runs/")))
		with open("runs/%s"%n) as file:
			result = yaml.load(file, Loader=yaml.UnsafeLoader)

		for t in range(result["experiment_params"]["T"]):
			r = {
				"t":t,
				"id":n.split("_")[1]
			}
			r.update(result["states"][t]['all_age_groups'])
			r.update(result["policy"][t]['all_age_groups'])
			results.append(r)


pd.DataFrame(results).to_csv("data.csv")
