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
        'mtest_cap' : 30000,
        'atest_cap' : 30000,
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




# @profile
def run_LPModel_no_intermVar(dynModel):
    # ##############################################################################
    # Testing the construction of a typical LP

    # shorthand for a few useful parameters
    T = dynModel.time_steps
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = 5 + 2*num_age_groups
    mixing_method = dynModel.mixing_method

    # uptimal decisions
    uopt_seq = np.zeros((ut_dim,T))

    # pick a starting u_hat sequence
    uhat_seq = np.zeros((ut_dim,T))
    # for now, homogenous testing
    Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
    uhat_seq[Nmtestg_idx_all,:] = dynModel.parameters['global-parameters']['C_mtest']/num_age_groups

    Natestg_idx_all = slice(controls.index('Natest_g'),ut_dim,num_controls)
    uhat_seq[Natestg_idx_all,:] = dynModel.parameters['global-parameters']['C_atest']/num_age_groups

    # and home lockdown variables all 1
    lock_home_idx_all = slice(controls.index('home'),ut_dim,num_controls)
    uhat_seq[lock_home_idx_all,:] = 1.0

    for k in range(T):
        M = gb.Model("Linearization Heuristic V2")
        M.Params.DualReductions = 0  # change this to get explicit infeasible or unbounded

        u_vars = {}
        x_vars = {}
        # ones_var =
        lock_home_idx_all = [controls.index('home') + num_controls * i for i in range(num_age_groups)]

        lb = np.zeros(ut_dim)
        for i in lock_home_idx_all:
            lb[i] = 1
        ub = np.ones(ut_dim) * float('inf')

        for i in range(num_age_groups):
            for act in range(4,10):
                ub[act + i * num_controls] = 1


        for ti in range(k,T):
            u_vars[ti] = M.addVars(ut_dim, ub=ub, lb=lb,  name="u_vars_time_{}".format(ti))

            x_vars[ti] = [0 for s in range(Xt_dim)]
        x_vars[T] =   [0 for s in range(Xt_dim)]

        M.update()


        Gamma_x, Gamma_u, K, all_labels = calculate_all_constraints(dynModel, bounce_existing)


        assert( np.shape(Gamma_x) == (num_constraints,Xt_dim) )
        assert( np.shape(Gamma_u) == (num_constraints,ut_dim) )
        assert( np.shape(K) == (num_constraints,T) )


        # Xhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq)


        Xhat_seq, new_uhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq, use_bounce_var)

        # print("Finished getting nominal trajectory for time {}".format(k))
        # print("-----------------------")

        assert( np.shape(Xhat_seq) == (Xt_dim,T-k) )
        assert( np.shape(new_uhat_seq) == (ut_dim,T-k) )

        # overwrite uhat with the updated one (with new bounce variables)


        uhat_seq = new_uhat_seq


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
                    x_vars[t][i] = Xhat_t[i]

            jacob_X = get_Jacobian_X(dynModel, Xhat_t, uhat_t, mixing_method)
            jacob_u = get_Jacobian_u(dynModel, Xhat_t, uhat_t, mixing_method)

            # Calculate linearization coefficients for X(t+1)
            At[t] = np.eye(Xt_dim) + dynModel.dt * jacob_X
            Bt[t] = dynModel.dt * jacob_u
            ct[t] = dynModel.dt * (get_F(dynModel, Xhat_t, uhat_t) - jacob_X @ Xhat_t - jacob_u @ uhat_t)


            for s in range(Xt_dim):

                state_values = gb.quicksum(At[t][s,i] * x_vars[t][i] for i in range(Xt_dim))
                control_values = gb.quicksum(Bt[t][s,i] * u_vars[t][i] for i in range(ut_dim))

                x_vars[t+1][s] = state_values + control_values + ct[t][s]

            for s in range(Xt_dim):
                M.addConstr(x_vars[t+1][s] >= 0)

            problemConst[t] = {}
            for c in range(len(Gamma_x)):

                problemConst[t][c] = M.addConstr(gb.quicksum(Gamma_x[c,s] *
                x_vars[t][s] for s in range(Xt_dim)) + gb.quicksum(Gamma_u[c,i] * u_vars[t][i] for i in range(ut_dim)) <= K[c,t], name="Const{}_time_{}".format(c,t))

            work_index = controls.index('work')
            transport_index = controls.index('transport')

            for i in range(num_age_groups):
                # print(f"Adding transport constraint for group {i} at time {k}")
                work_idx = work_index + i*num_controls
                transport_idx = transport_index + i*num_controls

                M.addConstr(u_vars[t][work_idx] * dynModel.transport_lb_work_fraction <= u_vars[t][transport_idx])



        # if k == 1:
        #     print(f"k is {k}")
        #     print(f"Xhat_seq is {Xhat_seq}")
        #     print(f"uhat_seq is {uhat_seq}")

        d, e = calculate_objective_time_dependent_coefs(dynModel, k, Xhat_seq, uhat_seq)



        M.setParam( 'OutputFlag', False )     # make Gurobi silent
        M.setParam( 'LogFile', "" )

        # print(f"Calculating objective at time {k}")
        terminalValue = gb.quicksum(d[s,T-k] * x_vars[T][s] for s in range(len(x_vars[T])))

        # print(f"Calculating interm values of objective at time {k}")
        # if k == 0:
        #     print(f"coeff of u{2}[0]: {e[0,2]}")
        #     print(f"coeff of u{2}[1]: {e[1,2]}")
        #     print(f"coeff of u{2}[2]: {e[2,2]}")
        #     print(f"coeff of u{2}[3]: {e[3,2]}")
        #     print(f"coeff of u{t}[5]: {e[5,t-k]}")
        #     print(f"coeff of u{t}[8]: {e[8,t-k]}")
        #     print(f"coeff of u{t}[15]: {e[15,t-k]}")
        #     print(f"coeff of u{t}[17]: {e[17,t-k]}")
        #     print(f"coeff of u{t}[18]: {e[18,t-k]}")
        #     print(f"coeff of u{t}[19]: {e[19,t-k]}")

            # M.addConstr(u_vars[2][0] == 30000)

        intermValues = gb.quicksum((gb.quicksum(e[i,ti-k] * u_vars[ti][i] for i in range(len(u_vars[ti]))) +  gb.quicksum(d[s,ti-k] * x_vars[ti][s] for s in range(len(x_vars[ti])))) for ti in range(k, T))

        # gb.quicksum(d[s,t-k] * x_vars[t][s] for s in range(len(x_vars[t]))) +
# gb.quicksum(e[i,t-k] * u_vars[t][i] for i in range(len(u_vars[t])))

        M.setObjective(intermValues + terminalValue , gb.GRB.MAXIMIZE)
# + terminalValue

        # print(d[:, T-k])

        # + d[:,T-k] @ x_vars[T]
        # M.setObjective(0)


        # print(f"Optimizing model for time {k}")
        M.optimize()
        # M.write(f"LP_model_NoIntermVar_k={k}.lp")

        print(f"Objective value for LPRef k = {k}: {M.objVal}")

        if( M.Status ==  gb.GRB.INFEASIBLE ):
            # model was infeasible
            M.computeIIS()  # irreducible system of infeasible inequalities
            M.write("LP_lineariz_IIS-V2.ilp")
            assert(False)

        # extract decisions for current period (testing and alphas)

        uopt_seq[:,k] = [u_vars[k][i].X for i in range(ut_dim)]
        uk_opt_dict, alphak_opt_dict = buildAlphaDict([u_vars[k][i].X for i in range(ut_dim)])

        m_tests = {}
        a_tests = {}
        BH = {}
        BICU = {}
        for ag in age_groups:
            BH[ag] = uk_opt_dict[ag]['BounceH_g']
            BICU[ag] = uk_opt_dict[ag]['BounceICU_g']
            m_tests[ag] = uk_opt_dict[ag]['Nmtest_g']
            a_tests[ag] = uk_opt_dict[ag]['Natest_g']

        # take one time step in dynamical system
        dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict, BH, BICU)

        # update uhat_sequence
        uhat_seq = np.zeros((ut_dim, T-k-1))
        for ti in range(k+1,T):
            uhat_seq[:,ti-k-1] = [u_vars[ti][i].X for i in range(ut_dim)]
        # print(f"u_optSeq at time {k} is {uopt_seq[:,k]}")
        # print(f"uhat_seq is {uhat_seq}")
        # print(f"States at stage {k}")
        # print(dynModel.get_state(k))

        # if k == 1:
        #     assert False
            # print(x_vars)
            # print([u_vars[t][i].X for i in range(ut_dim) for t in range(k,T)])


    pickle.dump(dynModel,open(f"../linearization_heuristic_dyn_models/TESTLPdynModel_NoIntermVar_linHeur_Prop_Bouncing_n_days={T}_deltas={dynModel.experiment_params['delta_schooling']}_xi={dynModel.experiment_params['xi']}_icus={dynModel.icus}_maxTests={dynModel.parameters['global-parameters']['C_atest']}.p","wb"))

    dynModel.print_stats()
    # print("uopt matrix is")
    # print(uopt_seq)



def run_LPModel_interm_X(dynModel):
    # ##############################################################################
    # Testing the construction of a typical LP

    # shorthand for a few useful parameters
    T = dynModel.time_steps
    Xt_dim = num_compartments * num_age_groups
    ut_dim = num_controls * num_age_groups
    num_constraints = 5 + 2*num_age_groups
    mixing_method = dynModel.mixing_method

    # uptimal decisions
    uopt_seq = np.zeros((ut_dim,T))

    # pick a starting u_hat sequence
    uhat_seq = np.zeros((ut_dim,T))
    # for now, homogenous testing
    Nmtestg_idx_all = slice(controls.index('Nmtest_g'),ut_dim,num_controls)
    uhat_seq[Nmtestg_idx_all,:] = dynModel.parameters['global-parameters']['C_mtest']/num_age_groups

    Natestg_idx_all = slice(controls.index('Natest_g'),ut_dim,num_controls)
    uhat_seq[Natestg_idx_all,:] = dynModel.parameters['global-parameters']['C_atest']/num_age_groups

    # and home lockdown variables all 1
    lock_home_idx_all = slice(controls.index('home'),ut_dim,num_controls)
    uhat_seq[lock_home_idx_all,:] = 1.0

    for k in range(T):
        M = gb.Model("Linearization Heuristic V2")
        M.Params.DualReductions = 0  # change this to get explicit infeasible or unbounded

        u_vars = {}
        x_vars = {}
        # ones_var =
        lock_home_idx_all = [controls.index('home') + num_controls * i for i in range(num_age_groups)]

        lb = np.zeros(ut_dim)
        for i in lock_home_idx_all:
            lb[i] = 1
        ub = np.ones(ut_dim) * float('inf')

        for i in range(num_age_groups):
            for act in range(4,10):
                ub[act + i * num_controls] = 1


        for ti in range(k,T):
            u_vars[ti] = M.addMVar(ut_dim, ub=ub, lb=lb,  name="u_vars_time_{}".format(ti))
            x_vars[ti] = M.addMVar(Xt_dim, name="x_vars_time_{}".format(ti))
        x_vars[T] = M.addMVar(Xt_dim, name="x_vars_time_{}".format(T))




        Gamma_x, Gamma_u, K, all_labels = calculate_all_constraints(dynModel, bounce_existing)

        # Xhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq)


        Xhat_seq, new_uhat_seq = get_X_hat_sequence(dynModel, k, uhat_seq, use_bounce_var)

        # print("Finished getting nominal trajectory for time {}".format(k))
        # print("-----------------------")

        assert( np.shape(Xhat_seq) == (Xt_dim,T-k) )
        assert( np.shape(new_uhat_seq) == (ut_dim,T-k) )

        # overwrite uhat with the updated one (with new bounce variables)
        #print("\nOld uhat at 1:")
        #print(uhat_seq[:,1])
        #print("\nNew uhat at 1")
        #print(new_uhat_seq[:,1])

        uhat_seq = new_uhat_seq

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

            # print(get_F(dynModel, Xhat_t, uhat_t)[116])
            # print((jacob_X @ Xhat_t)[116])
            #
            # print("BH_hat for group 8 at time {}: {}".format(t, uhat_t[8*num_controls+2]))
            # print((jacob_u @ uhat_t)[116])
            # print(ct[t][116])

            #Dynamic constraints binding x(t+1) with x(t) and u(t)
            dynamicConst[t] = M.addConstr(x_vars[t+1] == At[t] @ x_vars[t] + Bt[t] @ u_vars[t] + ct[t], name="Dynamics_const_time_{}".format(t))

            # for s in range(Xt_dim):
            #     state_values = sum(At[t][s,i] * x_vars[t][i] for i in range(Xt_dim))
            #     control_values = sum(Bt[t][s,i] * u_vars[t][i] for i in range(ut_dim))
            #
            #     M.addConstr(x_vars[t+1][s] == state_values + control_values + ct[t][s])

            # for c in range(len(Gamma_x)):
            #
            #     M.addConstr(sum(Gamma_x[c,s] *
            #     x_vars[t][s] for s in range(Xt_dim)) + sum(Gamma_u[c,i] * u_vars[t][i] for i in range(ut_dim)) <= K[c,t], name="Const{}_time_{}".format(c,t))




            # All constraints of the problem in matrix form
            problemConst[t] = M.addConstr(Gamma_x @ x_vars[t] + Gamma_u @ u_vars[t] <= K[:,t], name="All_const_time_{}".format(t))

            work_index = controls.index('work')
            transport_index = controls.index('transport')

            for i in range(num_age_groups):
                work_idx = work_index + i*num_controls
                transport_idx = transport_index + i*num_controls

                M.addConstr(u_vars[t][work_idx] * dynModel.transport_lb_work_fraction <= u_vars[t][transport_idx])


        # if k == 1:
        #     print(f"k is {k}")
        #     print(f"Xhat_seq is {Xhat_seq}")
        #     print(f"uhat_seq is {uhat_seq}")

        d, e = calculate_objective_time_dependent_coefs(dynModel, k, Xhat_seq, uhat_seq)

        M.setParam( 'OutputFlag', False )     # make Gurobi silent
        M.setParam( 'LogFile', "" )


        M.setObjective(sum(e[:,ti-k] @ u_vars[ti] + d[:,ti-k] @ x_vars[ti] for ti in range(k, T)) + d[:,T-k] @ x_vars[T], gb.GRB.MAXIMIZE)

        # e[:,t-k] @ u_vars[t]
        # d[:,t-k] @ x_vars[t]
        # + d[:,T-k] @ x_vars[T]

        # print(d[:, T-k])

        # + d[:,T-k] @ x_vars[T]
        # M.setObjective(0)




        M.optimize()
        # M.write(f"LP_model_IntermVar_k={k}.lp")

        print(f"Objective value for LPRef k = {k}: {M.objVal}")

        if( M.Status ==  gb.GRB.INFEASIBLE ):
            # model was infeasible
            M.computeIIS()  # irreducible system of infeasible inequalities
            M.write("LP_lineariz_IIS-V2.ilp")
            assert(False)

        # extract decisions for current period (testing and alphas)

        uopt_seq[:,k] = u_vars[k].X
        # print("uopt_k for k=",k,"is ",uopt_seq[:,k])
        uk_opt_dict, alphak_opt_dict = buildAlphaDict(u_vars[k].X)

        m_tests = {}
        a_tests = {}
        BH = {}
        BICU = {}
        for ag in age_groups:
            BH[ag] = uk_opt_dict[ag]['BounceH_g']
            BICU[ag] = uk_opt_dict[ag]['BounceICU_g']
            m_tests[ag] = uk_opt_dict[ag]['Nmtest_g']
            a_tests[ag] = uk_opt_dict[ag]['Natest_g']

        # take one time step in dynamical system
        dynModel.take_time_step(m_tests, a_tests, alphak_opt_dict, BH, BICU)

        # update uhat_sequence
        uhat_seq = np.zeros((ut_dim, T-k-1))
        for ti in range(k+1,T):
            uhat_seq[:,ti-k-1] = u_vars[ti].X

        # print(f"uhat_seq is {uhat_seq}")
        #
        # print(f"States at stage {k}")
        # print(dynModel.get_state(k))


    pickle.dump(dynModel,open(f"../linearization_heuristic_dyn_models/TESTLP_IntermVar_dynModel_linHeur_Prop_Bouncing_n_days={T}_deltas={dynModel.experiment_params['delta_schooling']}_xi={dynModel.experiment_params['xi']}_icus={dynModel.icus}_maxTests={dynModel.parameters['global-parameters']['C_atest']}.p","wb"))

    dynModel.print_stats()
    # print("uopt matrix is")
    # print(uopt_seq)



def main():

    region = "fitted"
    econ_param = "econ"
    # "econ"
    # "econ-death-zero"
    # "econ-zero"
    xi = 1e6

    start_time = time.time()
    for T in range(28,29,1):
        print(f"T is {T}")
        print("----------------")
        # print("LP Model with interm var")
        # dynModel = initializeDynModel(T, region, econ_param, xi)
        #
        # run_LPModel_interm_X(dynModel)
        # LPTotalReward = dynModel.get_total_reward()
        # dynModel.print_stats()
        # print(f"Total reward: {LPTotalReward}")

        #
        # end_time_LPModel = time.time()
        # print("----------------")
        # print("LP Model no interm var")
        # dynModel = initializeDynModel(T, region, econ_param, xi)
        #
        # run_LPModel_no_intermVar(dynModel)
        # LP_no_intermVar_TotalReward = dynModel.get_total_reward()
        #
        # assert abs(LP_no_intermVar_TotalReward-LPTotalReward)<0.000001, f"Both Total Rewards are not the same: LP_noIntermVar={LP_no_intermVar_TotalReward}, LPTotalReward={LPTotalReward}"


        print("----------------")
        print("Lin Huer")
        dynModel = initializeDynModel(T, region, econ_param, xi)

        run_heuristic_linearization(dynModel)
        dynModel.print_stats()
        LinHeurTotalReward = dynModel.get_total_reward()
        print("---------------")

        pickle.dump(dynModel,open(f"../linearization_heuristic_dyn_models/TESTLP_linHeur_Prop_Bouncing_n_days={T}_deltas={dynModel.experiment_params['delta_schooling']}_xi={dynModel.experiment_params['xi']}_icus={dynModel.icus}_maxTests={dynModel.parameters['global-parameters']['C_atest']}.p","wb"))


        # if abs(LinHeurTotalReward-LPTotalReward)>0.000001:
        #     print(f"Both Total Rewards are not the same: LinTotRew={LinHeurTotalReward}, LPTotalReward={LPTotalReward}")
        # end_time_lin = time.time()

    # print(f"Total time LP Model: {end_time_LPModel-start_time}")
    # print(f"Total time Lin Model: {end_time_lin-end_time_LPModel}")

if __name__ == "__main__":
    main()
