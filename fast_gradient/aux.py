import numpy as np
from copy import deepcopy

age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
#age_groups = ['all_age_groups']

cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]
all_activities = ['home','leisure','other','school','transport','work']
rel_activities = ['leisure','other','school','transport','work']

def state_to_matrix(state):
	m = np.zeros((len(age_groups),len(cont)), order="C")
	for i in range(len(age_groups)):
		for c in range(len(cont)):
			m[i,c] = state[age_groups[i]][cont[c]]
	return m


def matrix_to_alphas(m, freq):
	alphas_vec = []
	for t in range(m.shape[0]):
		alphas = {}
		for i in range(len(age_groups)):
			alphas[age_groups[i]] = {"home":1.0}
			for a in range(len(rel_activities)):
				alphas[age_groups[i]][rel_activities[a]] = float(m[t,i,a])
		for s in range(freq):
			alphas_vec.append(deepcopy(alphas))
	return alphas_vec

def alphas_to_matrix(alphas):
	m = np.zeros((len(age_groups),len(all_activities)))
	for i in range(len(age_groups)):
		for j in range(len(all_activities)):
			m[i,j] = alphas[age_groups[i]][all_activities[j]]
	return m

def matrix_to_vect_of_dict(v, freq):
	vec = []
	for t in range(v.shape[0]):
		d = {}
		for i in range(len(age_groups)):
			d[age_groups[i]] = float(v[t,i])
		for s in range(freq):
			vec.append(deepcopy(d))
	return vec

def matrix_to_state(m):
	state = {}
	for i,age_group in enumerate(age_groups):
		state[age_group] = {}
		for j,c in enumerate(cont):
			state[age_group][c] = m[i,j]
	return state

def dict_to_vector(d):
	v = []
	for ag in age_groups:
		v.append(d[ag])
	return np.array(v)

