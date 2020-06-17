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

# Define mixing method
mixing_method = {
    "name":"mult",
    "param_alpha":1.0,
    "param_beta":0.5,
    "param":1.0
    #"param":float(args.mixing_param) if args.mixing_param else 0.0,
}

# Read group parameters
with open("../parameters/"+simulation_params["region"]+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("../initialization/initialization.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    initialization = yaml.load(file, Loader=yaml.FullLoader)

# Define policy
with open('../benchmarks/static_infected_10.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    policy_file = yaml.load(file, Loader=yaml.FullLoader)
alphas_vec = policy_file['alphas_vec']

# Percentage infected at time 0
perc_infected = 10





# Initialize groups
random.seed(0)
for group in initialization:
	for chamber in initialization[group]:
		if chamber != "N":
			initialization[group][chamber] = random.randint(1000)
	initialization[group]["N"] = initialization[group]["S"] + initialization[group]["E"] + initialization[group]["I"] + initialization[group]["R"]

# Randomly assign alphas_vec
for group in alphas_vec[0]:
	for activity in alphas_vec[0][group]:
		alphas_vec[0][group][activity] = random.uniform(0,1)


u = random.randint(1000, size = (num_controls * num_age_groups))
u_hat_dict, alphas = buildAlphaDict(u)

m_tests = {}
a_tests = {}
for g in age_groups:
    m_tests[g] = u_hat_dict[g]['Nmtest_g']
    a_tests[g] = u_hat_dict[g]['Natest_g']

# Construct model
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
initial_state = dynModel.get_state(0)

# Run model for one step
for i in range(1):
	dynModel.take_time_step(m_tests, a_tests, alphas_vec[0])
final_X = dynModel.get_state(1)

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

state_matrix = state_to_matrix(initial_state)
m_tests_vec = tests_to_vector(m_tests)
a_tests_vec = tests_to_vector(a_tests)
alphas_matrix = alphas_to_matrix(alphas_vec[0])

fast = FastDynamicalModel(universe_params, simulation_params['dt'], mixing_method)
new_state, econs = fast.take_time_step(state_matrix, m_tests_vec, a_tests_vec, alphas_matrix)


slow_total_contacts = np.zeros(len(age_groups), order="C")
for i,group in enumerate(age_groups):
	slow_total_contacts[i] = dynModel.groups[group].total_contacts[0]

assert(np.linalg.norm(fast.total_contacts-slow_total_contacts)<1e-9)

print(new_state-state_to_matrix(final_X))


### Now we time the models

# Run model for many steps
def main1():
	dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
	initial_state = dynModel.get_state(0)

	t0 = time.time()
	for i in range(200):
		dynModel.take_time_step(m_tests, a_tests, alphas_vec[0])
	final_X = dynModel.get_state(200)
	t1 = time.time()
	print(t1-t0)

# Same thing

fast = FastDynamicalModel(universe_params, simulation_params['dt'], mixing_method)

def main2():
	state = state_to_matrix(initial_state)

	t0 = time.time()
	for i in range(200):
		if (i%14 == 0):
			update_contacts = True
		else:
			update_contacts = False
		state, econs = fast.take_time_step(state, m_tests_vec, a_tests_vec, alphas_matrix, update_contacts=update_contacts)
	t1 = time.time()
	print(t1-t0)

# cProfile.run('main1()')
# cProfile.run('main2()')

# Now we test the models for a larger horizon
# Construct model
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)
initial_state = dynModel.get_state(0)

# Run model for one step
iters = 10
for i in range(iters):
	dynModel.take_time_step(m_tests, a_tests, alphas_vec[0])
final_X = dynModel.get_state(iters)
total_reward = np.sum(dynModel.rewards[1:])


state_matrix = state_to_matrix(initial_state)
m_tests_vec = tests_to_vector(m_tests)
a_tests_vec = tests_to_vector(a_tests)
alphas_matrix = alphas_to_matrix(alphas_vec[0])

fast = FastDynamicalModel(universe_params, simulation_params['dt'], mixing_method)
state = state_matrix
fast_total_reward = 0
for i in range(iters):
	state, econs = fast.take_time_step(state, m_tests_vec, a_tests_vec, alphas_matrix)
	fast_total_reward += econs["reward"]

print("Norm of matrix difference:", np.linalg.norm((state-state_to_matrix(final_X))/state_to_matrix(final_X)))
# print((state-state_to_matrix(final_X))/state_to_matrix(final_X))
# print(state_to_matrix(final_X))
print("Difference of rewards:",(total_reward-fast_total_reward)/total_reward)






