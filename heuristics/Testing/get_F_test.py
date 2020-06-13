import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,grandparentdir) 


import math
import yaml
import random

from linearization import get_F, X_to_dict, dict_to_X, buildAlphaDict, dict_to_u, get_Jacobian_X, get_Jacobian_u
from group import *
from numpy import random


age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29','age_group_30_39', 'age_group_40_49', 'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
SEIR_groups = [ 'S_g', 'E_g', 'I_g', 'R_g', 'N_g', 'Ia_g', 'Ips_g', \
       'Ims_g', 'Iss_g', 'Rq_g', 'H_g', 'ICU_g', 'D_g' ]
activities = ['home','leisure','other','school','transport','work']
controls = [ 'Nmtest_g', 'Natest_g', 'BounceH_g', 'BounceICU_g' ]
controls.extend(activities)
print(controls)

num_age_groups = len(age_groups)
num_compartments = len(SEIR_groups)
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
    "name":"maxmin",
    "param_alpha":1.0,
    "param_beta":0.5,
    "param":1.0
    #"param":float(args.mixing_param) if args.mixing_param else 0.0,
}

# Read group parameters
with open("../../parameters/"+simulation_params["region"]+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("../../initialization/initialization.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    initialization = yaml.load(file, Loader=yaml.FullLoader)

# Define policy
with open('../../benchmarks/static_infected_10.yaml') as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    policy_file = yaml.load(file, Loader=yaml.FullLoader)
alphas_vec = policy_file['alphas_vec']

# Percentage infected at time 0
perc_infected = 10
# Randomly assign people to groups

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




dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)

initial_state = dynModel.get_state(0)
initial_X = dict_to_X(initial_state)

##### Test X_to_dict and dict_to_X
new_state = X_to_dict(dict_to_X(initial_state))
for group in initial_state:
	for chamber in initial_state[group]:
		assert(new_state[group][chamber] == initial_state[group][chamber])

X_state = random.randint(1000, size = (num_compartments * num_age_groups))
new_X_state = dict_to_X(X_to_dict(X_state))
assert(np.all(X_state == new_X_state))

#### Test the buildAlphaDict and dict_to_u
u = random.randint(1000, size = (num_controls * num_age_groups))
u_hat_dict, alphas = buildAlphaDict(u)
new_u = dict_to_u(u_hat_dict, alphas)
assert(np.all(u == new_u))

u = dict_to_u(u_hat_dict, alphas_vec[0])
new_u_hat_dict, new_alphas = buildAlphaDict(u)
for group in new_alphas:
	for activity in new_alphas[group]:
		assert(new_alphas[group][activity] == alphas_vec[0][group][activity])
	for control in new_u_hat_dict[group]:
		assert(new_u_hat_dict[group][control] == u_hat_dict[group][control])

### Test the getF function, we start by transitioning 100 times


# Run it once
m_tests = {}
a_tests = {}
for g in age_groups:
    m_tests[g] = u_hat_dict[g]['Nmtest_g']
    a_tests[g] = u_hat_dict[g]['Natest_g']


for i in range(100):
	dynModel.take_time_step(m_tests, a_tests, alphas_vec[0])

final_X = dict_to_X(dynModel.get_state(100))

# Run it again
for g in age_groups:
	u_hat_dict[g]['BounceH_g'] = False
	u_hat_dict[g]['BounceICU_g'] = False
X = initial_X
u = dict_to_u(u_hat_dict, alphas_vec[0])

for i in range(100):
	X = get_F(dynModel,X,u)

assert(np.all(X == final_X))
assert(dynModel.t == 100)

### We test by transitioning first 100 times
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)

X = initial_X
for i in range(100):
	X = get_F(dynModel,X,u)

for i in range(100):
	dynModel.take_time_step(m_tests, a_tests, alphas_vec[0])
final_X = dict_to_X(dynModel.get_state(100))

assert(np.all(X == final_X))
assert(dynModel.t == 100)


## Now we test with bouncing variables
iterations = 50
u = random.randint(1,1000, size = (num_controls * num_age_groups))
u_hat_dict, alphas = buildAlphaDict(u)

dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)

initial_X = dict_to_X(initial_state)

# Run it once
m_tests = {}
a_tests = {}
for g in age_groups:
    m_tests[g] = u_hat_dict[g]['Nmtest_g']
    a_tests[g] = u_hat_dict[g]['Natest_g']

B_H = {}
B_ICU = {}
for g in age_groups:
    B_H[g] = u_hat_dict[g]['BounceH_g'] if (u_hat_dict[g]['BounceH_g'] != -1) else False
    B_ICU[g] = u_hat_dict[g]['BounceICU_g'] if (u_hat_dict[g]['BounceICU_g'] != -1) else False

for i in range(iterations):
	dynModel.take_time_step(m_tests, a_tests, alphas_vec[0], B_H, B_ICU)

final_X = dict_to_X(dynModel.get_state(iterations))

# Run it again
X = initial_X
u = dict_to_u(u_hat_dict, alphas_vec[0])
for i in range(iterations):
	X = get_F(dynModel,X,u)

assert(np.all(X == final_X))
assert(dynModel.t == iterations)






# Now we check that the Jacobian with respect to X is calculated properly
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)


X = initial_X
u = random.randint(1,1000, size = (num_controls * num_age_groups))
u_hat_dict, alphas = buildAlphaDict(u)
u = dict_to_u(u_hat_dict, alphas_vec[0])

epsilon = 1e-6

numerical_jacobian = np.zeros((num_age_groups*num_compartments, num_age_groups*num_compartments))
for i in range(num_age_groups*num_compartments):
	one = np.zeros(num_age_groups*num_compartments)
	F = get_F(dynModel,initial_X,u)
	one[i] = F[i]*epsilon
	Fdt = get_F(dynModel,initial_X+one,u)
	
	partial = (Fdt - F)/(epsilon*F[i])
	numerical_jacobian[:,i] = partial

real_jacobian = get_Jacobian_X(dynModel, initial_X, u, mixing_method) + np.identity(num_age_groups*num_compartments)

distance = (numerical_jacobian-real_jacobian)
scaled = np.zeros((num_age_groups*num_compartments, num_age_groups*num_compartments))
for i in range(num_age_groups*num_compartments):
	for j in range(num_age_groups*num_compartments):
		scaled[i,j] = np.abs(distance[i,j]/real_jacobian[i,j]) if real_jacobian[i,j]!=0 else np.abs(distance[i,j])

print(np.max(scaled))
print(np.where(scaled==np.amax(scaled)))

# Now the Jacobian with respect to u
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)


X = initial_X
u = random.randint(1,1000, size = (num_controls * num_age_groups))
u_hat_dict, alphas = buildAlphaDict(u)
u = dict_to_u(u_hat_dict, alphas_vec[0])


epsilon = 1e-5

numerical_jacobian = np.zeros((num_age_groups*num_compartments,num_age_groups*num_controls))
for i in range(num_age_groups*num_controls):
	one = np.zeros(num_age_groups*num_controls)

	F = get_F(dynModel,initial_X,u)
	one[i] = u[i]*epsilon
	Fdt = get_F(dynModel,initial_X,u+one)
	partial = (Fdt - F)/(epsilon*u[i])
	numerical_jacobian[:,i] = partial


real_jacobian = get_Jacobian_u(dynModel, initial_X, u, mixing_method)

distance = (numerical_jacobian-real_jacobian)
scaled = np.zeros((num_age_groups*num_compartments,num_age_groups*num_controls))
for i in range(num_age_groups*num_compartments):
	for j in range(num_age_groups*num_controls):
		scaled[i,j] = np.abs(distance[i,j]/real_jacobian[i,j]) if real_jacobian[i,j]!=0 else np.abs(distance[i,j])

print(np.max(scaled))
print(np.where(scaled==np.amax(scaled)))


