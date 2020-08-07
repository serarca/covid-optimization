import os.path
import sys
from inspect import getsourcefile
from random import *

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/heuristics")

from linearization import *
from heuristics import *

# Global variables
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
# Move population to infected (without this there is no epidem.)
for group in initialization:
	change = initialization[group]["S"]*perc_infected/100
	initialization[group]["S"] = initialization[group]["S"] - change
	initialization[group]["I"] = initialization[group]["I"] + change

# Create environment
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method)

# Set up testing decisions: no testing for now
a_tests_vec, m_tests_vec = no_tests(dynModel)
tests = {
    'a_tests_vec':a_tests_vec,
    'm_tests_vec':m_tests_vec,
}

dynModel.simulate(m_tests_vec, a_tests_vec, alphas_vec)


# # ############# Testing X_hat_sequence #####################
# #
# # # Obtain a random sequence of tests to compare the X_hat seq with
# # # the dynamics of the model
# max_a_tests = [10e4 for i in range(dynModel.time_steps)]
# max_m_tests = [10e4 for i in range(dynModel.time_steps)]
# groups = []
# for group in dynModel.parameters['seir-groups']:
# 	population = sum([dynModel.initialization[group][sg] for sg in ["S","E","I","R","Ia","Ips","Ims","Iss","Rq","H","ICU","D"]])
# 	if population > 0:
# 		groups.append(group)
# groups.sort()
#
# random_a_test_vec, random_m_test_vec = random_partition(dynModel, groups, max_a_tests, max_m_tests)
#
# for k in range(dynModel.time_steps-1, 0, -1):
#     # Construct u_hat sequence
#     u_hat_sequence = np.zeros((dynModel.time_steps-k, num_controls * num_age_groups))
#
#     for i in range(dynModel.time_steps - k):
#         for ag in range(num_age_groups):
#             u_hat_sequence[i, ag * num_controls + controls.index('Nmtest_g')] = m_tests_vec[k+i][age_groups[ag]]
#
#             u_hat_sequence[i, ag * num_controls + controls.index('Natest_g')] = a_tests_vec[k+i][age_groups[ag]]
#
#             for act in activities:
#                 u_hat_sequence[i, ag * num_controls + controls.index(act)] = alphas_vec[k+i][age_groups[ag]][act]
#
#     #Get X_hat_sequence (which also runs the dyn model from k to the end)
#     X_hat_sequence = get_X_hat_sequence(dynModel, k, u_hat_sequence)
#
#     #Check that the states of the dyn model are the same as X_hat
#
#     for i in range(dynModel.time_steps - k):
#         for ag in range(num_age_groups):
#             for st in SEIR_groups:
#                 # The assertion checks that for each compartment,
#                 # at each time and for each group, the values in
#                 # X_hat_sequence coincide with the values in the
#                 # states of dynModel. st is the state/
#                 # compartment, and we get the names used un group
#                 # by removing the _g.
#
#                 assert(
#                 X_hat_sequence[i, ag * num_compartments + SEIR_groups.index(st)]
#                 ==
#                 dynModel.get_state(k+i)[age_groups[ag]][st.replace('_g','')])
#
# print("Finished succesfully the equality test for X_hat_sequence. It coincides with the states in dynModel every time.")
#
# ###########################
# # Test the constraint coefficients by generating random
# # values for the parameters and then checking that they
# # coincide with the definitions.
#
# for n,g in dynModel.groups.items():
#     g.parameters['mu'] = uniform(0.1,10)
#     g.parameters['lambda_H_D'] = uniform(0.1,10)
#     g.parameters['lambda_H_R'] = uniform(0.1,10)
#     g.parameters['lambda_ICU_R'] = uniform(0.1,10)
#     g.parameters['lambda_ICU_D'] = uniform(0.1,10)
#     g.parameters['p_ICU'] = uniform(0.1,10)
#     g.parameters['p_H'] = uniform(0.1,10)
#
# a, b = calculate_ICU_coefficients(dynModel)
# assert(a.shape == (1, num_age_groups * num_compartments))
# assert(b.shape == (1, num_age_groups * num_controls))
#
# for ag in range(num_age_groups):
#     assert(
#             b[0, ag * num_controls + controls.index('BounceICU_g')]
#             ==
#             -1
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('I_g')]
#             ==
#             dynModel.groups[age_groups[ag]].parameters['mu'] * dynModel.groups[age_groups[ag]].parameters['p_ICU']
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('Iss_g')]
#             ==
#             dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_ICU'] / (dynModel.groups[age_groups[ag]].parameters['p_ICU'] + dynModel.groups[age_groups[ag]].parameters['p_H']))
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('ICU_g')]
#             ==
#             (1 - dynModel.groups[age_groups[ag]].parameters['lambda_ICU_R'] - dynModel.groups[age_groups[ag]].parameters['lambda_ICU_D'])
#           )
#     assert(np.count_nonzero(a) == 3 * num_age_groups)
#     assert(np.count_nonzero(b) == num_age_groups)
#
#
#
# a, b = calculate_H_coefficients(dynModel)
# assert(a.shape == (1, num_age_groups * num_compartments))
# assert(b.shape == (1, num_age_groups * num_controls))
#
# for ag in range(num_age_groups):
#     assert(
#             b[0, ag * num_controls + controls.index('BounceH_g')]
#             ==
#             -1
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('I_g')]
#             ==
#             dynModel.groups[age_groups[ag]].parameters['mu'] * dynModel.groups[age_groups[ag]].parameters['p_H']
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('Iss_g')]
#             ==
#             dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_H'] / (dynModel.groups[age_groups[ag]].parameters['p_ICU'] + dynModel.groups[age_groups[ag]].parameters['p_H']))
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('H_g')]
#             ==
#             (1 - dynModel.groups[age_groups[ag]].parameters['lambda_H_R'] - dynModel.groups[age_groups[ag]].parameters['lambda_H_D'])
#           )
#     assert(np.count_nonzero(a) == 3 * num_age_groups)
#     assert(np.count_nonzero(b) == num_age_groups)
#
#
# for ag in range(num_age_groups):
#     a, b = calculate_BH_coefficients(dynModel, age_groups[ag])
#     assert(a.shape == (1, num_age_groups * num_compartments))
#     assert(b.shape == (1, num_age_groups * num_controls))
#
#
#     assert(
#             b[0, ag * num_controls + controls.index('BounceH_g')]
#             ==
#             1
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('I_g')]
#             ==
#             - dynModel.groups[age_groups[ag]].parameters['mu'] * dynModel.groups[age_groups[ag]].parameters['p_H']
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('Iss_g')]
#             ==
#             - dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_H'] / (dynModel.groups[age_groups[ag]].parameters['p_ICU'] + dynModel.groups[age_groups[ag]].parameters['p_H']))
#           )
#
#     assert(np.count_nonzero(a) == 2)
#     assert(np.count_nonzero(b) == 1)
#
#
# for ag in range(num_age_groups):
#     a, b = calculate_BICU_coefficients(dynModel, age_groups[ag])
#     assert(a.shape == (1, num_age_groups * num_compartments))
#     assert(b.shape == (1, num_age_groups * num_controls))
#     assert(
#             b[0, ag * num_controls + controls.index('BounceICU_g')] == 1
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('I_g')]
#             ==
#             - dynModel.groups[age_groups[ag]].parameters['mu'] * dynModel.groups[age_groups[ag]].parameters['p_ICU']
#           )
#
#     assert(
#             a[0, ag * num_compartments + SEIR_groups.index('Iss_g')]
#             ==
#             - dynModel.groups[age_groups[ag]].parameters['mu'] * (dynModel.groups[age_groups[ag]].parameters['p_ICU'] / (dynModel.groups[age_groups[ag]].parameters['p_ICU'] + dynModel.groups[age_groups[ag]].parameters['p_H']))
#           )
#
#     assert(np.count_nonzero(a) == 2)
#     assert(np.count_nonzero(b) == 1)
# print("All tests for the coefficients succesfully passed.")
