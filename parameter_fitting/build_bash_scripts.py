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

days_to_try = [d*2 for d in range(30,50)]
alphas_to_try = [(a+1)*0.15 for a in range(0,25)]

open('sherlock_master_0.sh', 'w').close()
counter = 0
for days in days_to_try:
	for alphas in alphas_to_try:
		open('sherlock_scripts_0/script_%d.sh'%counter, 'w').close()
		with open('sherlock_scripts_0/script_%d.sh'%counter, 'a') as the_file:
			the_file.write(
		 		"""#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 ParameterFittingRandomToPython.py --days %d --alpha %f --scenario 0
"""%(days,alphas
								)
							)

		with open('sherlock_master_0.sh', 'a') as the_file:
			the_file.write("sbatch sherlock_scripts_0/script_%d.sh\n"%counter)
		counter += 1




open('sherlock_master_1.sh', 'w').close()
counter = 0
for days in days_to_try:
	for alphas in alphas_to_try:
		open('sherlock_scripts_1/script_%d.sh'%counter, 'w').close()
		with open('sherlock_scripts_1/script_%d.sh'%counter, 'a') as the_file:
			the_file.write(
		 		"""#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 ParameterFittingRandomToPython.py --days %d --alpha %f --scenario 1
"""%(days,alphas
								)
							)

		with open('sherlock_master_1.sh', 'a') as the_file:
			the_file.write("sbatch sherlock_scripts_1/script_%d.sh\n"%counter)
		counter += 1
