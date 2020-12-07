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



# Parameters to try
with open("../parameters/run_params.yaml") as file:
	run_params = yaml.load(file, Loader=yaml.FullLoader)

params_to_try = run_params["params_to_try"]
groups = run_params["groups"]

open('parallel_bash.sh', 'w').close()
for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					for eta in params_to_try["eta"]:
						with open('parallel_bash.sh', 'a') as the_file:
						    the_file.write(
						    	'python simple_benchmarks.py --delta %f --icus %d --eta %f --groups all --xi %f --a_tests %d --m_tests %d &\n'%(
						    		delta, icus, eta, xi, tests[1], tests[0]
						    	)
						    )

for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					for eta in params_to_try["eta"]:
						with open('parallel_bash.sh', 'a') as the_file:
						    the_file.write(
						    	'python constant_gradient_benchmarks.py --delta %f --icus %d --eta %f --groups all --xi %f --a_tests %d --m_tests %d &\n'%(
						    		delta, icus, eta, xi, tests[1], tests[0]
						    	)
						    )

for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					for eta in params_to_try["eta"]:
						with open('parallel_bash.sh', 'a') as the_file:
						    the_file.write(
						    	'python dynamic_gradient_benchmarks.py --delta %f --icus %d --eta %f --groups all --xi %f --a_tests %d --m_tests %d &\n'%(
						    		delta, icus, eta, xi, tests[1], tests[0]
						    	)
						    )

# Now write the sherlock scripts

counter = 0
open('sherlock_master_1.sh', 'w').close()
open('sherlock_master_2.sh', 'w').close()
open('sherlock_master_remaining.sh', 'w').close()

for delta in params_to_try["delta_schooling"]:
	for xi in params_to_try["xi"]:
		for icus in params_to_try["icus"]:
			for tests in params_to_try["tests"]:
				for testing in params_to_try["testing"]:
					for eta in params_to_try["eta"]:
						open('sherlock_scripts/script_%d.sh'%counter, 'w').close()
						with open('sherlock_scripts/script_%d.sh'%counter, 'a') as the_file:
							the_file.write(
						 		"""#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 French_trigger_benchmark_ref --delta %f --icus %d --eta %f --groups all --xi %f --a_tests %d --m_tests %d
python3 ICU_admissions_trigger_benchmark_ref.py --delta %f --icus %d --eta %f --groups all --xi %f --a_tests %d --m_tests %d
"""%(
									delta, icus, eta, xi, tests[1], tests[0],
									delta, icus, eta, xi, tests[1], tests[0],
								)
							)
						if counter < 540:
							with open('sherlock_master_1.sh', 'a') as the_file:
								the_file.write("sbatch sherlock_scripts/script_%d.sh\n"%counter)
						else:
							with open('sherlock_master_2.sh', 'a') as the_file:
								the_file.write("sbatch sherlock_scripts/script_%d.sh\n"%counter)
						filename = "results/dynamic_gradient/xi-%d_icus-%d_testing-homogeneous_natests-%d_nmtests-%d_T-90_startday-60_groups-all_dschool-%f_eta-%f_freq-90-14.yaml"%(
							xi,icus,tests[1],tests[0],delta,eta
						)
						if (not path.exists(filename)):
							with open('sherlock_master_remaining.sh', 'a') as the_file:
								the_file.write("sbatch sherlock_scripts/script_%d.sh\n"%counter)


						counter+=1






