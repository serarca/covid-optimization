import yaml
from inspect import getsourcefile
import os.path
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
from os import path

days_to_try = [d*2 for d in range(25,50)]
alphas_to_try = [a*0.1 for a in range(20,30)]

open('sherlock_master.sh', 'w').close()
counter = 0
for days in days_to_try:
	for alphas in alphas_to_try:
		open('sherlock_scripts/script_%d.sh'%counter, 'w').close()
		with open('sherlock_scripts/script_%d.sh'%counter, 'a') as the_file:
			the_file.write(
		 		"""#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 ParameterFittingRandomToPython.py --days %d --alpha %f
"""%(days,alphas
								)
							)

		with open('sherlock_master.sh', 'a') as the_file:
			the_file.write("sbatch sherlock_scripts/script_%d.sh\n"%counter)
		counter += 1
