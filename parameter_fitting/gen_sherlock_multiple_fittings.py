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



# Now write the sherlock scripts

open('sherlock_master_multiple_fittings.sh', 'w').close()

for identifier in range(14):
	open('sherlock_multiple_fittings/script_%d.sh'%identifier, 'w').close()
	with open('sherlock_multiple_fittings/script_%d.sh'%identifier, 'a') as the_file:
		the_file.write(
	 		"""#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
ml gurobi
python3 ProcessMultipleFittings.py --identifier %d
"""%(identifier)
		)

	with open('sherlock_master_multiple_fittings.sh', 'a') as the_file:
		the_file.write("sbatch sherlock_multiple_fittings/script_%d.sh\n"%identifier)







