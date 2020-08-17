import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 


import math
import yaml
from group import *
from numpy import random
from fast_group import FastDynamicalModel
import time
import cProfile


# Convert alphas and testing and state to matrices
def state_to_matrix(state):
	m = np.zeros((len(age_groups),len(cont)), order="C")
	for i in range(len(age_groups)):
		for c in range(len(cont)):
			m[i,c] = state[age_groups[i]][cont[c]]
	return m

def tests_to_vector(tests):
	v = np.zeros(len(age_groups), order="C")
	for i in range(len(age_groups)):
		v[i] = tests[age_groups[i]]
	return v

def alphas_to_matrix(alphas):
	m = np.zeros((len(age_groups),len(activities)), order="C")
	for i in range(len(age_groups)):
		for a in range(len(activities)):
			m[i,a] = alphas[age_groups[i]][activities[a]]
	return m

def state_to_matrix(state_dict):
	m = np.zeros((len(age_groups), len(cont)), order="C")
	for i in range(len(age_groups)):
		for j in range(len(cont)):
			m[i,j] = state_dict[age_groups[i]][cont[j]]
	return m

def buildAlphaDict(u_hat_array):
    """Given an array u_hat_array, builds a dictionary for all decisions that is compatible with DynModel"""
    u_hat_dict = {}
    alphas = {}
    for ag in range(0, num_age_groups):
        u_hat_dict[age_groups[ag]] = {}
        alphas[age_groups[ag]] = {}
        u_hat_dict[age_groups[ag]]['Nmtest_g'] = u_hat_array[ag * num_controls + controls.index('Nmtest_g')]
        u_hat_dict[age_groups[ag]]['Natest_g'] = u_hat_array[ag * num_controls + controls.index('Natest_g')]
        u_hat_dict[age_groups[ag]]['BounceH_g'] = u_hat_array[ag * num_controls + controls.index('BounceH_g')] if u_hat_array[ag * num_controls + controls.index('BounceH_g')] != -1 else False
        u_hat_dict[age_groups[ag]]['BounceICU_g'] = u_hat_array[ag * num_controls + controls.index('BounceICU_g')] if u_hat_array[ag * num_controls + controls.index('BounceICU_g')] != -1 else False

        alphas[age_groups[ag]]['home'] = u_hat_array[ag * num_controls + controls.index('home')]
        alphas[age_groups[ag]]['leisure'] = u_hat_array[ag * num_controls + controls.index('leisure')]
        alphas[age_groups[ag]]['other'] = u_hat_array[ag * num_controls + controls.index('other')]
        alphas[age_groups[ag]]['school'] = u_hat_array[ag * num_controls + controls.index('school')]
        alphas[age_groups[ag]]['transport'] = u_hat_array[ag * num_controls + controls.index('transport')]
        alphas[age_groups[ag]]['work'] = u_hat_array[ag * num_controls + controls.index('work')]

    return u_hat_dict, alphas

age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29','age_group_30_39', 'age_group_40_49', 'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]
activities = ['home','leisure','other','school','transport','work']
controls = [ 'Nmtest_g', 'Natest_g', 'BounceH_g', 'BounceICU_g' ]
controls.extend(activities)

num_age_groups = len(age_groups)
num_controls = len(controls)
num_activities = len(activities)

simulation_params = {
        'dt':1.0,
        'days': 182,
        'region': "Ile-de-France",
        'quar_freq': 182,
}


# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))

# Read group parameters
with open("../parameters/fitted.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("../initialization/60days.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    initialization = yaml.load(file, Loader=yaml.FullLoader)
    start_day = 61

# Read econ parameters
with open("../parameters/econ.yaml") as file:
	econ_params = yaml.load(file, Loader=yaml.FullLoader)


experiment_params = {
	'delta_schooling':0.5,
	'xi':0,
	'icus':2000,
}

# Define mixing parameter
mixing_method = universe_params["mixing"]




# Randomly assign alphas_vec
alphas = {}
random.seed(0)
for group in age_groups:
	alphas[group] = {}
	for activity in activities:
		alphas[group][activity] = random.uniform(0,1)


u = random.randint(1000, size = (num_controls * num_age_groups))
u_hat_dict, _ = buildAlphaDict(u)

m_tests = {}
a_tests = {}
for g in age_groups:
    m_tests[g] = u_hat_dict[g]['Nmtest_g']
    a_tests[g] = u_hat_dict[g]['Natest_g']

# Construct model
dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day)

initial_state = dynModel.get_state(0)

# Run model for one step
for i in range(1):
	econs_slow = dynModel.take_time_step(m_tests, a_tests, alphas)
final_X = dynModel.get_state(1)



state_matrix = state_to_matrix(initial_state)
m_tests_vec = tests_to_vector(m_tests)
a_tests_vec = tests_to_vector(a_tests)
alphas_matrix = alphas_to_matrix(alphas)

fast = FastDynamicalModel(universe_params, econ_params, experiment_params, simulation_params['dt'], mixing_method)
for t in range(1):
	if start_day + t < universe_params['days_before_gamma']:
			lockdown_status = "pre-gamma"
	else:
		lockdown_status = "post-gamma"
	print(lockdown_status)
	new_state, econs_fast = fast.take_time_step(state_matrix, m_tests_vec, a_tests_vec, alphas_matrix, lockdown_status)


slow_total_contacts = np.zeros(len(age_groups), order="C")
for i,group in enumerate(age_groups):
	slow_total_contacts[i] = dynModel.groups[group].total_contacts[0]

assert(np.linalg.norm(fast.total_contacts-slow_total_contacts)<1e-9)

print(new_state-state_to_matrix(final_X))

print(econs_fast)
print({"economic_value":econs_slow["economic_value"],
	"deaths":econs_slow["deaths"],
	"reward":econs_slow["reward"]})

# ### Now we time the models

# # Run model for many steps
# def main1():
# 	dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
# 	initial_state = dynModel.get_state(0)

# 	t0 = time.time()
# 	for i in range(200):
# 		dynModel.take_time_step(m_tests, a_tests, alphas_vec[0])
# 	final_X = dynModel.get_state(200)
# 	t1 = time.time()
# 	print(t1-t0)

# # Same thing

# fast = FastDynamicalModel(universe_params, simulation_params['dt'], mixing_method)

# def main2():
# 	state = state_to_matrix(initial_state)

# 	t0 = time.time()
# 	for i in range(200):
# 		if (i%14 == 0):
# 			update_contacts = True
# 		else:
# 			update_contacts = False
# 		state, econs = fast.take_time_step(state, m_tests_vec, a_tests_vec, alphas_matrix, update_contacts=update_contacts)
# 	t1 = time.time()
# 	print(t1-t0)

# # cProfile.run('main1()')
# # cProfile.run('main2()')

# # Now we test the models for a larger horizon
# Construct model
dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, start_day)
initial_state = dynModel.get_state(0)

# Run model for one step
iters = 100
sum_reward = 0
sum_deaths = 0
sum_econs = 0
for i in range(iters):
	econs_slow = dynModel.take_time_step(m_tests, a_tests, alphas)
	sum_reward += econs_slow['reward']
	sum_deaths += econs_slow['deaths']
	sum_econs += econs_slow['economic_value']
final_X = dynModel.get_state(iters)
print({"economic_value":sum_reward,
	"deaths":sum_deaths,
	"reward":sum_econs})

state_matrix = state_to_matrix(initial_state)
m_tests_vec = tests_to_vector(m_tests)
a_tests_vec = tests_to_vector(a_tests)
alphas_matrix = alphas_to_matrix(alphas)

fast = FastDynamicalModel(universe_params, econ_params, experiment_params, simulation_params['dt'], mixing_method)
state = state_matrix
fast_sum_reward = 0
fast_sum_deaths = 0
fast_sum_econs = 0
for t in range(iters):
	if start_day + t < universe_params['days_before_gamma']:
			lockdown_status = "pre-gamma"
	else:
		lockdown_status = "post-gamma"
	state, econs_fast = fast.take_time_step(state, m_tests_vec, a_tests_vec, alphas_matrix, lockdown_status)
	fast_sum_reward += econs_fast['reward']
	fast_sum_deaths += econs_fast['deaths']
	fast_sum_econs += econs_fast['economic_value']
print({"economic_value":fast_sum_reward,
	"deaths":fast_sum_deaths,
	"reward":fast_sum_econs})

print("Norm of matrix difference:", np.linalg.norm((state-state_to_matrix(final_X))/state_to_matrix(final_X)))
# print((state-state_to_matrix(final_X))/state_to_matrix(final_X))
# print(state_to_matrix(final_X))
#print("Difference of rewards:",(total_reward-fast_total_reward)/total_reward)






