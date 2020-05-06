import random
import numpy as np
from collections import defaultdict

# A heuristic that assigns all testing to a given group
def all_to_one(dyn_model, max_a_tests, max_m_tests):
	# Choose a group randomly
	group = random.choice(dyn_model.groups.keys())
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

	return (a_tests,m_tests)


# A heuristic that assigns random testing at each point in time
def random_partition(dyn_model, max_a_tests, max_m_tests):
	# Choose a group randomly
	a_sample = defaultdict(list)
	m_sample = defaultdict(list)
	# Sample dictionary of A tests for all groups at all times uniformly from the simplex boundary
	for t in range(dyn_model.time_steps):
		sample_sum = 0
		for n in dyn_model.groups:
			sample = np.random.uniform()
			a_sample[n].append(sample)
			sample_sum += sample
		for n in dyn_model.groups:
			a_sample[n][t] = a_sample[n][t]/sample_sum*max_a_tests[t]
	# Sample dictionary of M tests for all groups at all times uniformly from the simplex boundary
	for t in range(dyn_model.time_steps):
		sample_sum = 0
		for n in dyn_model.groups:
			sample = np.random.uniform()
			m_sample[n].append(sample)
			sample_sum += sample
		for n in dyn_model.groups:
			m_sample[n][t] = m_sample[n][t]/sample_sum*max_m_tests[t]

	return (a_sample,m_sample)