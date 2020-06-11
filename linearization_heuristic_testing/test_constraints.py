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
        'days': 3,
        'region': "Ile-de-France",
        'quar_freq': 1,
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
#print(dynModel.time_steps)

# Set up testing decisions: no testing for now
a_tests_vec, m_tests_vec = no_tests(dynModel)
tests = {
    'a_tests_vec':a_tests_vec,
    'm_tests_vec':m_tests_vec,
}

# add parameters for testing capacity
dynModel.parameters['global-parameters']['C_mtest'] = 10000
dynModel.parameters['global-parameters']['C_atest'] = 10000

# ##############################################################################
# Testing the construction of a typical LP

# shorthand for a few useful parameters
T = dynModel.time_steps
Xt_dim = num_compartments * num_age_groups
ut_dim = num_controls * num_age_groups
num_constraints = 4 + 2*num_age_groups + num_age_groups*num_activities

# uptimal decisions
uopt_seq = np.zeros((ut_dim,T))

uhat_seq = np.zeros((ut_dim,T))

for k in range(T):
    M = gb.Model("Linearization Heuristic V2")
    M.Params.DualReductions = 0  # change this to get explicit infeasible or unbounded

    u_vars = {}
    x_vars = {}
    # ones_var =
    for ti in range(k,T):
        u_vars[ti] = M.addMVar(ut_dim, name="u_vars_time_{}".format(ti))
        x_vars[ti] = M.addMVar(Xt_dim, name="x_vars_time_{}".format(ti))
    x_vars[T] = M.addMVar(Xt_dim, name="x_vars_time_{}".format(T))




    Gamma_x, Gamma_u, K, all_labels = calculate_all_constraints(dynModel)

    Xhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq)

    d, e = calculate_objective_time_dependent_coefs(dynModel, k, Xhat_seq, uhat_seq)

    # M.setObjective(sum(d[:,t-k] @ x_vars[t] + e[:,t-k] @ u_vars[t] for t in range(k, T-1)) + d[:,T-k-1] @ x_vars[T], gb.GRB.MAXIMIZE)

    M.setObjective(0)

    At = {}
    Bt = {}
    ct = {}

    dynamicConst = {}
    problemConst = {}
    #for t in range(k,T):
    for t in range(k,T):
        # get Xhat(t) and uhat(t)
        Xhat_t = Xhat_seq[:,t-k]
        uhat_t = uhat_seq[:,t-k]

        #Initial conditions
        if t == k:
            for i in range(Xt_dim):
                M.addConstr(Xhat_t[i] == x_vars[t][i], name="Initial Conditions")

        jacob_X = get_Jacobian_X(dynModel, Xhat_t, uhat_t, mixing_method)
        jacob_u = get_Jacobian_u(dynModel, Xhat_t, uhat_t, mixing_method)

        # Calculate linearization coefficients for X(t+1)
        At[t] = np.eye(Xt_dim) + dynModel.dt * jacob_X
        Bt[t] = dynModel.dt * jacob_u
        ct[t] = dynModel.dt * (get_F(dynModel, Xhat_t, uhat_t) - jacob_X @ Xhat_t - jacob_u @ uhat_t)

        print(get_F(dynModel, Xhat_t, uhat_t)[116])
        print((jacob_X @ Xhat_t)[116])

        print("BH_hat for group 8 at time {}: {}".format(t, uhat_t[8*num_controls+2]))
        print((jacob_u @ uhat_t)[116])
        print(ct[t][116])

        #Dynamic constraints binding x(t+1) with x(t) and u(t)
        dynamicConst[t] = M.addConstr(x_vars[t+1] == At[t] @ x_vars[t] + Bt[t] @ u_vars[t] + ct[t], name="Dynamics_const_time_{}".format(t))

        # All constraints of the problem in matrix form
        problemConst[t] = M.addConstr(Gamma_x @
        x_vars[t] + Gamma_u @ u_vars[t] <= K[:,t], name="All_const_time_{}".format(t))

    M.optimize()

    if( M.Status ==  gb.GRB.INFEASIBLE ):
        # model was infeasible
        M.computeIIS()  # irreducible system of infeasible inequalities
        M.write("LP_lineariz_IIS-V2.ilp")
        assert(False)

    # extract decisions for current period (testing and alphas)
    for i in range(num_age_groups):
        print(u_vars[k].X[i* num_controls + 2])
    uopt_seq[:,k] = u_vars[k].X
    uk_opt_dict, alphak_opt_dict = buildAlphaDict(u_vars[k].X)

    m_tests = {}
    a_tests = {}
    for ag in age_groups:
        m_tests[ag] = uk_opt_dict[ag]['Nmtest_g']
        a_tests[ag] = uk_opt_dict[ag]['Natest_g']

    # take one time step in dynamical system
    dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict)

    # update uhat_sequence
    uhat_seq = np.zeros((ut_dim, T-k-1))
    for t in range(k+1,T):
        uhat_seq[:,t-k-1] = u_vars[t].X















##############################################
#
# for k in range(T):
#     M = gb.Model("Linearization Heuristic V2")
#     M.Params.DualReductions = 0  # change this to get explicit infeasible or unbounded
#
#     u_vars = {}
#     x_vars = {}
#     # ones_var =
#     for ti in range(k,T):
#         u_vars[ti] = M.addMVar(ut_dim, name="u_vars_time_{}".format(ti))
#         x_vars[ti] = np.ones(Xt_dim)
#         # x_vars[ti] = M.addMVar(Xt_dim, name="x_vars_time_{}".format(ti))
#     # x_vars[T] = M.addMVar(Xt_dim, name="x_vars_time_{}".format(T))
#
#     M.setObjective(0)
#
#
#     Gamma_x, Gamma_u, K, all_labels = calculate_all_constraints(dynModel)
#
#     Xhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq)
#
#     At = {}
#     Bt = {}
#     ct = {}
#
#     dynamicConst = {}
#     realConst = {}
#     #for t in range(k,T):
#     for t in range(k,T):
#         # get Xhat(t) and uhat(t)
#         Xhat_t = Xhat_seq[:,t-k]
#         uhat_t = uhat_seq[:,t-k]
#
#         #Initial conditions
#         if t == k:
#             for i in range(Xt_dim):
#                 x_vars[t] = Xhat_t
#
#         jacob_X = get_Jacobian_X(dynModel, Xhat_t, uhat_t, mixing_method)
#         jacob_u = get_Jacobian_u(dynModel, Xhat_t, uhat_t, mixing_method)
#
#         # Calculate linearization coefficients for X(t+1)
#         At[t] = np.eye(Xt_dim) + dynModel.dt * jacob_X
#         Bt[t] = dynModel.dt * jacob_u
#         ct[t] = dynModel.dt * (get_F(dynModel, Xhat_t, uhat_t) - jacob_X @ Xhat_t - jacob_u @ uhat_t)
#
#         print(t)
#         print(x_vars[t])
#         print(At[t] @ x_vars[t])
#
#         x_vars[t+1] = At[t] @ x_vars[t] + Bt[t] @ u_vars[t] + ct[t]
#
#         realConst[t] = M.addConstr(Gamma_x @ x_vars[t] + Gamma_u @ u_vars[t] <= K[:,t], name="All_const_time_{}".format(t))
#
#     M.optimize()
#
#     if( M.Status ==  gb.GRB.INFEASIBLE ):
#         # model was infeasible
#         M.computeIIS()  # irreducible system of infeasible inequalities
#         M.write("LP_lineariz_IIS-V2.ilp")
#         assert(False)
#
#     # extract decisions for current period (testing and alphas)
#
#     uopt_seq[:,k] = u_vars[k].X
#     uk_opt_dict, alphak_opt_dict = buildAlphaDict(u_vars[k].X)
#
#     m_tests = {}
#     a_tests = {}
#     for ag in age_groups:
#         m_tests[ag] = uk_opt_dict[ag]['Nmtest_g']
#         a_tests[ag] = uk_opt_dict[ag]['Natest_g']
#
#     # take one time step in dynamical system
#     dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict)
#
#     # update uhat_sequence
#     uhat_seq = np.zeros((ut_dim, T-k-1))
#     for t in range(k+1,T):
#         uhat_seq[:,t-k-1] = u_vars[t].X
