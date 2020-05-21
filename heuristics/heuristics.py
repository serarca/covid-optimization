import random
import numpy as np
from collections import defaultdict
# Import group module
from inspect import getsourcefile
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
from group import SEIR_group, DynamicalModel

def change_order(d):
	d_array = []
	times = len(d[list(d.keys())[0]])
	for t in range(times):
		curr_d = {
			key:d[key][t] for key in d
		}
		d_array.append(curr_d)
	return d_array

def re_change_order(l):
	times = len(l)
	groups = l[0].keys()
	d = {}
	for g in groups:
		d[g] = [test[g] for test in l]

	return d

# A heuristic that assigns all testing to a given group
def all_to_one(dyn_model, group, max_a_tests, max_m_tests):
    # Choose a group randomly
    print("Chose group '%s' to give all testing"%group)
    a_tests = {}
    m_tests = {}
    for name in dyn_model.groups:
        if name == group:
            a_tests[name] = max_a_tests[:]
            m_tests[name] = max_m_tests[:]
        else:
            a_tests[name] = [0 for t in range(dyn_model.time_steps)]
            m_tests[name] = [0 for t in range(dyn_model.time_steps)]

    return (change_order(a_tests),change_order(m_tests))


# A heuristic that assigns random testing at each point in time among all groups in 'groups' variable
def random_partition(dyn_model, groups, max_a_tests, max_m_tests):

	a_sample = defaultdict(list)
	m_sample = defaultdict(list)
	# Sample dictionary of A tests for all groups at all times uniformly from the simplex boundary
	for t in range(dyn_model.time_steps):
		sample_sum = 0
		for n in dyn_model.groups:

			if n in groups:
				sample = np.random.uniform()
			else:
				sample = 0
			a_sample[n].append(sample)
			sample_sum += sample
		for n in dyn_model.groups:
			a_sample[n][t] = a_sample[n][t] / sample_sum * max_a_tests[t]

	# Sample dictionary of M tests for all groups at all times uniformly from the simplex boundary
	for t in range(dyn_model.time_steps):
		sample_sum = 0
		for n in dyn_model.groups:
			if n in groups:
				sample = np.random.uniform()
			else:
				sample = 0
			m_sample[n].append(sample)
			sample_sum += sample
		for n in dyn_model.groups:
			m_sample[n][t] = m_sample[n][t]/sample_sum*max_m_tests[t]

	return (change_order(a_sample),change_order(m_sample))

# A heuristic that divides testing homogeneously among all groups in groups variable
def homogeneous(dyn_model, max_a_tests, max_m_tests):
	groups = dyn_model.groups
	a_tests = {}
	m_tests = {}
	for name in dyn_model.groups:
		if name in groups:
			a_tests[name] = [max_a_tests[i]/(len(groups)+0.0) for i in range(len(max_a_tests))]
			m_tests[name] = [max_m_tests[i]/(len(groups)+0.0) for i in range(len(max_m_tests))]
		else:
			a_tests[name] = [0 for t in range(len(max_m_tests))]
			m_tests[name] = [0 for t in range(len(max_m_tests))]

	return (change_order(a_tests),change_order(m_tests))

# A heuristic that assigns all testing to a given group
def no_tests(dyn_model):
	# Choose a group randomly
	a_tests = {}
	m_tests = {}
	for name in dyn_model.groups:
		a_tests[name] = [0 for t in range(dyn_model.time_steps)]
		m_tests[name] = [0 for t in range(dyn_model.time_steps)]

	return (change_order(a_tests),change_order(m_tests))
