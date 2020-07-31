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
import pickle
import math
import yaml
import time

# Global variables

bounce_existing = False
use_bounce_var = True



def initializeDynModel(T=5, region="Ile-de-France", econ_param="econ", xi=0):
    # simulation_params = {
    #         'dt':1.0,
    #         'days': 30,
    #         'region': "fitted",
    #         'quar_freq': 1,
    # }

    # Define time variables


    # Define mixing method
    simulation_params = {
        'dt':1.0,
        'region': region,
        'quar_freq': 1,
        'num_days' : T,
        'initial_infected_count' : 1,
        'perc_infected' : 10,
        'mixing_method' : {
            "name":"mult",
            "param_alpha":1.0,
            "param_beta":0.5,},
        'mtest_cap' : 0,
        'atest_cap' : 0,
        'work_full_lockdown_factor' : 0.24,
        'heuristic': 'linearization',
        'transport_lb_work_fraction': 0.25
    }

    simulation_params['time_periods'] = int(math.ceil(simulation_params["num_days"]/simulation_params["dt"]))
    # Read group parameters
    with open("../parameters/"+simulation_params["region"]+".yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        universe_params = yaml.load(file, Loader=yaml.FullLoader)

    # Read initialization
    with open("../initialization/fitted.yaml") as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        initialization = yaml.load(file, Loader=yaml.FullLoader)

    with open(f"../parameters/{econ_param}.yaml") as file:
        econ_params = yaml.load(file, Loader=yaml.FullLoader)

    # Define policy
    # with open('../benchmarks/static_infected_10.yaml') as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        # policy_file = yaml.load(file, Loader=yaml.FullLoader)
    # alphas_vec = policy_file['alphas_vec']

    # Percentage infected at time 0
    # perc_infected = 10
    # Move population to infected (without this there is no epidem.)
    # for group in initialization:
    # 	change = initialization[group]["S"]*perc_infected/100
    # 	initialization[group]["S"] = initialization[group]["S"] - change
    # 	initialization[group]["I"] = initialization[group]["I"] + change

    delta = 0.5

    # 30 * 37199.03
    icus = 2000
    tests_freq = 7
    lockdown_freq = 14

    experiment_params = {
        'delta_schooling':delta,
        'xi':xi,
        'icus':icus,
        'test_freq': tests_freq,
        'lockdown_freq': lockdown_freq
    }
    num_time_periods = int(math.ceil(simulation_params["num_days"]/simulation_params["dt"]))

    # Create environment
    dynModel = DynamicalModel(universe_params, econ_params, experiment_params, initialization, simulation_params['dt'], num_time_periods, simulation_params['mixing_method'], simulation_params['transport_lb_work_fraction'])
    #print(dynModel.time_steps)

    # Set up testing decisions: no testing for now
    a_tests_vec, m_tests_vec = no_tests(dynModel)
    tests = {
        'a_tests_vec':a_tests_vec,
        'm_tests_vec':m_tests_vec,
    }

    # add parameters for testing capacity
    dynModel.parameters['global-parameters']['C_mtest'] = simulation_params['mtest_cap']
    dynModel.parameters['global-parameters']['C_atest'] = simulation_params['atest_cap']

    return dynModel


def linearization_errors(dynModel, k, u_hat_seq, u_seq):
    ''' Input: Dynamical model, starting point k, u_hat_seq, u_seq. The controls u_hat_seq and u_seq should be a numpy array of dim (ut_dim, T-k). The input u_seq denotes the controls for the periods from k to T-1, while the u_hat_seq is the nominal trajectory around which we linearize.
    Output: Prints to console the error in the total number of deaths at time T from the linearized dynamic around u_hat_seq.'''

    T = dynModel.time_steps
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = 5 + 2*num_age_groups
    mixing_method = dynModel.mixing_method


    x_vars = np.zeros((Xt_dim,T-k+1))

    X_seq, new_u_seq = get_X_hat_sequence(dynModel, k, u_seq, use_bounce_var)

    Xhat_seq, new_uhat_seq = get_X_hat_sequence(dynModel, k, u_hat_seq, use_bounce_var)

    # print("Finished getting nominal trajectory for time {}".format(k))
    # print("-----------------------")

    assert( np.shape(Xhat_seq) == (Xt_dim,T-k+1) )
    assert( np.shape(new_uhat_seq) == (ut_dim,T-k) )

    # overwrite uhat with the updated one (with new bounce variables)
    #print("\nOld uhat at 1:")
    #print(uhat_seq[:,1])
    #print("\nNew uhat at 1")
    #print(new_uhat_seq[:,1])

    for t in range(T-k):
        for i in range(ut_dim):
            if abs(u_hat_seq[i,t] - new_uhat_seq[i,t]) > 0.0001:
                print(f"Original value: {u_hat_seq[i,t]}")
                print(f"New value: {new_uhat_seq[i,t]}")
                print(f"t is {t}")
                print(f"index is {i}")

    At = {}
    Bt = {}
    ct = {}

    #for t in range(k,T):
    for t in range(k,T):
        # get Xhat(t) and uhat(t)
        Xhat_t = Xhat_seq[:,t-k]
        uhat_t = new_uhat_seq[:,t-k]

        #Initial conditions
        if t == k:
            x_vars[:,t-k] = Xhat_t

        jacob_X = get_Jacobian_X(dynModel, Xhat_t, uhat_t, mixing_method)
        jacob_u = get_Jacobian_u(dynModel, Xhat_t, uhat_t, mixing_method)

        # Calculate linearization coefficients for X(t+1)
        At[t] = np.eye(Xt_dim) + dynModel.dt * jacob_X
        Bt[t] = dynModel.dt * jacob_u
        ct[t] = dynModel.dt * (get_F(dynModel, Xhat_t, uhat_t) - jacob_X @ Xhat_t - jacob_u @ uhat_t)


        #Dynamic constraints binding x(t+1) with x(t) and u(t)
        x_vars[:, t+1-k] = At[t] @ x_vars[:, t-k] + Bt[t] @ u_seq[:, t-k] + ct[t]

        # infected_idx =

        print(f"t = {t}")
        print("real dynamics Exposed for time t+1:")
        print(X_seq[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments) , t+1-k])
        print("linearized dynamics for exposed:")
        print(x_vars[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments) , t+1-k])

        print("At[t] @ x[t] in the exposed indices:")
        print((At[t] @ x_vars[:, t-k])[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments)])
        print("Bt[t] @ u_seq[t] in the exposed indices:")
        print((Bt[t] @ u_seq[:, t-k])[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments)])


        print("C[t] in the exposed indices is:")
        print(ct[t][slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments)])
        print("get_F in the exposed indices is:")
        print(get_F(dynModel, Xhat_t, uhat_t)[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments)])
        print("- jacob_X @ Xhat in the exposed indices is:")
        print((- jacob_X @ Xhat_t)[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments)])
        print("- jacob_u @ uhat in the exposed indices is:")
        print((- jacob_u @ uhat_t)[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments)])
        print("jacob u in row E:")
        print(jacob_u[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments),:])

        print(f"u hat at t is: {uhat_t}")



        assert (x_vars[slice(SEIR_groups.index('E_g'), Xt_dim, num_compartments) , t+1-k] >= 0).all()



        # print("real dynamics Infected for tine t+1:")
        # print(X_seq[slice(SEIR_groups.index('I_g'), Xt_dim, num_compartments) , t+1-k])
        # print("linearized dynamics for infected:")
        # print(x_vars[slice(SEIR_groups.index('I_g'), Xt_dim, num_compartments) , t+1-k])
        # print("C[t] in the infected indices is:")
        # print(ct[t][slice(SEIR_groups.index('I_g'), Xt_dim, num_compartments)])
        # print("At[t] @ x[t] in the infected indices:")
        # print((At[t] @ x_vars[:, t-k])[slice(SEIR_groups.index('I_g'), Xt_dim, num_compartments)])
        # print("Bt[t] @ u_seq[t] in the infected indices:")
        # print((Bt[t] @ u_seq[:, t-k])[slice(SEIR_groups.index('I_g'), Xt_dim, num_compartments)])


        if not (x_vars[slice(SEIR_groups.index('I_g'), Xt_dim, num_compartments) , t+1-k] >= 0).all():
            print(x_vars[slice(SEIR_groups.index('I_g'), Xt_dim, num_compartments) , t+1-k])
            assert(False)

        D_g_idxs = [ag * num_compartments + SEIR_groups.index('D_g') for ag in range(num_age_groups)]

        # print(f"Total cumulative deaths for linearized dynamics at time {t+1}: {sum(x_vars[i, t+1-k] for i in D_g_idxs)}")

    D_g_idxs = [ag * num_compartments + SEIR_groups.index('D_g') for ag in range(num_age_groups)]

    total_deaths_linearized = sum(x_vars[i, T-k] for i in D_g_idxs)



    # assert((new_u_seq==u_seq).all())

    total_deaths_real_dynamics = sum(X_seq[i, T-k] for i in D_g_idxs)

    print(f"The total error in final death count is: {total_deaths_real_dynamics - total_deaths_linearized}")
    print(f"Total deaths with real dynamics: {total_deaths_real_dynamics}")
    print(f"Total deaths with linearized dynamics: {total_deaths_linearized}")



def main():

    region = "fitted"
    econ_param = "econ-zero"
    # "econ"
    # "econ-death-zero"
    # "econ-zero"
    xi = 1e6


    T = 90
    dynModel = initializeDynModel(T, region, econ_param, xi)

    ## OBS: We could run the errors for other ks, but we'd need to run the dynamical model up to k
    k = 0

    # print(f"Error with nominal trajectory of full lockdown and no tests and evaluated point of full opening and homogenous testing.")
    # # Nominal Trajectory: No tests and everything closed down
    # u_hat_seq = np.zeros((ut_dim, T))
    #
    # # Sequence to be tested: Full openning up for the lockdowns, and homogenous testing
    # u_seq = np.zeros((ut_dim,T))
    #
    # # for now, homogenous testing
    # Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
    # u_seq[Nmtestg_idx_all,:] = dynModel.parameters['global-parameters']['C_mtest']/num_age_groups
    #
    # Natestg_idx_all = slice(controls.index('Natest_g'),ut_dim,num_controls)
    # u_seq[Natestg_idx_all,:] = dynModel.parameters['global-parameters']['C_atest']/num_age_groups
    #
    # # Starting the uhat_seq with all lockdowns set to 1 (fully open)
    #
    # for act in activities:
    #     act_indices = slice(controls.index(act), ut_dim, num_controls)
    #     u_seq[act_indices,:] = 1.0
    #
    # X, adjusted_u_seq = get_X_hat_sequence(dynModel, k, u_seq, use_bounce_var)
    #
    # X, adjusted_u_hat_seq = get_X_hat_sequence(dynModel, k, u_hat_seq, use_bounce_var)
    #
    # linearization_errors(dynModel, k, adjusted_u_hat_seq, adjusted_u_seq)

# ##############################################


    dynModel = initializeDynModel(T, region, econ_param, xi)

    print(f"Error with nominal trajectory of full opening and hom tests and evaluated point of full opening and homogenous testing.")
    # Nominal Trajectory: No tests and Full opening
    u_hat_seq = np.zeros((ut_dim,T))

    # for now, homogenous testing
    # Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
    # u_hat_seq[Nmtestg_idx_all,:] = dynModel.parameters['global-parameters']['C_mtest']/num_age_groups
    #
    # Natestg_idx_all = slice(controls.index('Natest_g'),ut_dim,num_controls)
    # u_hat_seq[Natestg_idx_all,:] = dynModel.parameters['global-parameters']['C_atest']/num_age_groups

    # Starting the uhat_seq with all lockdowns set to 1 (fully open)

    for act in activities:
        act_indices = slice(controls.index(act), ut_dim, num_controls)
        u_hat_seq[act_indices,:] = 1.0

    act_indices = slice(controls.index('home'), ut_dim, num_controls)
    u_hat_seq[act_indices,:] = 1.0

    # Sequence to be tested: Full openning up for the lockdowns, and homogenous testing
    u_seq = np.zeros((ut_dim,T))

    # for now, homogenous testing
    Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
    u_seq[Nmtestg_idx_all,:] = dynModel.parameters['global-parameters']['C_mtest']/num_age_groups

    Natestg_idx_all = slice(controls.index('Natest_g'),ut_dim,num_controls)
    u_seq[Natestg_idx_all,:] = dynModel.parameters['global-parameters']['C_atest']/num_age_groups

    # Starting the uhat_seq with all lockdowns set to 1 (fully open)

    for act in activities:
        act_indices = slice(controls.index(act), ut_dim, num_controls)
        u_seq[act_indices,:] = 0.0

    act_indices = slice(controls.index('home'), ut_dim, num_controls)
    u_seq[act_indices,:] = 1.0

    # assert((u_hat_seq == u_seq).all())
    # u_hat_seq = u_seq
    # for t in range(T-k):
    #     for i in range(ut_dim):
    #         u_hat_seq[i,t] = u_seq[i,t]
    #
    # u_hat_seq = np.copy(u_seq)

    X, adjusted_u_seq = get_X_hat_sequence(dynModel, k, u_seq, use_bounce_var)

    X, adjusted_u_hat_seq = get_X_hat_sequence(dynModel, k, u_hat_seq, use_bounce_var)

    linearization_errors(dynModel, k, adjusted_u_hat_seq, adjusted_u_seq)





if __name__ == "__main__":
    main()
