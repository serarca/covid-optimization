
import time
from gurobipy import *
from heuristics import *
import os.path
import sys
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
from group import SEIR_group, DynamicalModel, n_contacts

def forecasting_heuristic(dynModel, max_a_tests, max_m_tests, alphas, h_cap_vec, icu_cap_vec, tolerance, max_iterations, death_value, mixing_method):

    start_time = time.time()
    #Create copy of dyn model to modify
    mixing_method = {
    "name":"mult",
    "param":0.0,
    }
    start = time.time()
    dynModelC = DynamicalModel(dynModel.parameters, dynModel.initialization, dynModel.dt, dynModel.time_steps, mixing_method)

    #Initialize real testing vectors
    final_a_testing = {}
    final_m_testing = {}
    for g in dynModelC.groups:
        final_a_testing[g] = []
        final_m_testing[g] = []

    #Initialize old and new forecasting to zeros except for the first elements
    old_forecasting = {}
    new_forecasting = {}

    time_steps = range(dynModelC.time_steps)

    initializeForecastings(old_forecasting, new_forecasting, dynModelC)

    #For all times time_steps
    for t in time_steps:
        time_step_time = time.time()
        # Set the correct time steps
        remaining_time_steps = range(dynModel.time_steps - t)

        # Empty the copy of the dyn model and put as initial conditions the first elements of the old forecast.
        dynModelC.__init__(dynModelC.parameters, dynModelC.initialization, dynModelC.dt, len(remaining_time_steps), mixing_method)

        initialize_with_forecast(dynModelC, old_forecasting)

        # Run the model with no testing and fix the results as the old forecasting
        # Fix the new forecast as all all zeros
        # no_m_tests = {}
        # no_a_tests = {}
        # for g in dynModelC.groups:
        #     no_m_tests[g] = [0 for i in remaining_time_steps]
        #     no_a_tests[g] = [0 for i in remaining_time_steps]

        groups = []
        for group in dynModelC.parameters['seir-groups']:
        	population = sum([dynModelC.initialization[group][sg] for sg in ["S","E","I","R","Ia","Ips","Ims","Iss","Rq","H","ICU","D"]])
        	if population > 0:
        		groups.append(group)
        groups.sort()

        forecast_m_tests, forecast_a_tests = random_partition(dynModelC, groups, max_a_tests, max_m_tests)

        dynModelC.simulate(forecast_m_tests, forecast_a_tests, alphas)

        assign_forecastings(dynModelC, old_forecasting)

        iterations = 0

        #Create G Model (so as to modify it later)
        M = Model()
        start_building_var_time = time.time()

        B_ICU = M.addVars(remaining_time_steps, dynModelC.groups.keys(), vtype=GRB.CONTINUOUS, name="ICU-Bounces")
        B_H = M.addVars(remaining_time_steps,dynModelC.groups.keys(), vtype=GRB.CONTINUOUS, name="H-Bounces")
        A_test = M.addVars(remaining_time_steps, dynModelC.groups.keys(), vtype=GRB.CONTINUOUS, ub=max_a_tests[0], name="a-tests")
        M_test = M.addVars(remaining_time_steps, dynModelC.groups.keys(), vtype=GRB.CONTINUOUS, ub=max_m_tests[0], name="m-tests")

        end_building_var_time = time.time()

        # print("Total time building var: {}".format(end_building_var_time - start_building_var_time))

        M.update()

        #While true do (will break only when the number of iterations have completed or the tolerance level has been reached)
        while True:

            I_vector = {}
            for name, group in dynModelC.groups.items():
                I_vector[name] = [float('-inf') for ti in remaining_time_steps]

            #Write gurobi problem with fixed states to be the old forecast and obtain a seq of m and a tests
            #Objective
            start_obj_time = time.time()

            infected_contact_prob_vec = [obtain_Infected_contact_prob(dynModelC, M_test, A_test, old_forecasting, group, ti, alphas, I_vector, mixing_method) for ti in remaining_time_steps]

            end_inf_cont_prob_time = time.time()
            # print("Time spend getting inf cont prob: {}".format(end_inf_cont_prob_time - start_obj_time))


            economic_obj_E = quicksum(
            group.economics['work_value']*(
            alphas[ti][name]['work']+
            group.economics['lockdown_fraction']*(1-alphas[ti][name]['work'])
            ) * obtain_E(dynModelC, M_test, A_test, old_forecasting, group, ti, alphas, I_vector, mixing_method, infected_contact_prob_vec) for ti in remaining_time_steps for name, group in dynModelC.groups.items())

            end_econ_E_time = time.time()
            # print("Time spend getting E: {}".format(end_econ_E_time - end_inf_cont_prob_time))

            economic_obj_S = quicksum(
            group.economics['work_value']*(
            alphas[ti][name]['work']+
            group.economics['lockdown_fraction']*(1-alphas[ti][name]['work'])
            ) * obtain_S(dynModelC, M_test, A_test, old_forecasting, group, ti, alphas, I_vector, mixing_method, infected_contact_prob_vec) for ti in remaining_time_steps for name, group in dynModelC.groups.items())

            end_econ_S_time = time.time()
            # print("Time spend getting S: {}".format(end_econ_S_time - end_econ_E_time))

            economic_obj_R = quicksum(
            group.economics['work_value']*(
            alphas[ti][name]['work']+
            group.economics['lockdown_fraction']*(1-alphas[ti][name]['work'])
            ) * obtain_R(M_test, A_test, old_forecasting, group, ti, I_vector)  for ti in remaining_time_steps for name, group in dynModelC.groups.items())

            end_econ_R_time = time.time()
            # print("Time spend getting R: {}".format(end_econ_R_time - end_econ_S_time))

            economic_obj_Rq = quicksum( group.economics['work_value'] * obtain_Rq(M_test, A_test, old_forecasting, group, ti) for ti in remaining_time_steps for name, group in dynModelC.groups.items())

            end_econ_Rq_time = time.time()
            # print("Time spend getting Rq: {}".format(end_econ_Rq_time - end_econ_R_time))

            economic_obj = economic_obj_E + economic_obj_R + economic_obj_S + economic_obj_Rq

            end_econ_obj_time = time.time()

            # print("Time Summing up the econ obj: {}".format(end_econ_obj_time - end_econ_Rq_time))

            # print("Time building econ obj:  {}".format(end_econ_obj_time - start_obj_time))

            deaths = quicksum(death_value * obtain_Deaths(M_test, A_test, B_H, B_ICU, old_forecasting, group, remaining_time_steps, I_vector) for name, group in dynModelC.groups.items())

            end_death_cost_time = time.time()

            # print("Time building death cost: {}".format(end_death_cost_time - end_econ_obj_time))


            obj = economic_obj - deaths



            end_gen_obj_time = time.time()

            # print("Time joining both objectives: {}".format(end_gen_obj_time - end_death_cost_time))

            # print("Time building obj: {}".format(end_gen_obj_time - start_obj_time))

            M.setObjective(obj, GRB.MAXIMIZE)

            end_obj_time = time.time()

            # print("Total time adding obj to Gurobi: {}".format(end_obj_time - end_gen_obj_time))


            start_const_time = time.time()

            M.addConstrs((quicksum(group.parameters['mu'] * (group.parameters['p_H'] * obtain_I(M_test, A_test, old_forecasting, group, ti, I_vector))
            + (group.parameters['p_H']/((group.parameters['p_ICU'] + group.parameters['p_H']) if (group.parameters['p_ICU'] + group.parameters['p_H']) != 0 else 10e-6))
            * obtain_Iss(M_test,A_test,old_forecasting,group,ti) - B_H[ti, group.name] for name, group in dynModelC.groups.items()) <= h_cap_vec[ti] - quicksum((1 - group2.parameters['lambda_H_R'] - group2.parameters['lambda_H_D']) * get_H(M_test, A_test, B_H, B_ICU,  old_forecasting, group2, ti, I_vector) for name2, group2 in dynModelC.groups.items()) for ti in remaining_time_steps), name="HCAP")

            end_HCAP_const_time = time.time()
            # print("Time spend getting HCAP const: {}".format(end_HCAP_const_time - start_const_time))

            M.addConstrs((quicksum(group.parameters['mu'] * (group.parameters['p_ICU'] * obtain_I(M_test, A_test, old_forecasting, group, ti, I_vector)) + (group.parameters['p_ICU']
            /((group.parameters['p_ICU'] + group.parameters['p_H']) if (group.parameters['p_ICU'] + group.parameters['p_H']) != 0 else 10e-6))
            * obtain_Iss(M_test,A_test,old_forecasting,group,ti) - B_H[ti, group.name] for name, group in dynModelC.groups.items()) <= h_cap_vec[ti] - quicksum((1 - group2.parameters['lambda_ICU_R'] - group2.parameters['lambda_ICU_D']) * get_ICU(M_test, A_test, B_H, B_ICU,  old_forecasting, group2, ti, I_vector) for name2, group2 in dynModelC.groups.items()) for ti in remaining_time_steps), name="ICUCAP")

            end_ICUCAP_const_time = time.time()
            # print("Time spend getting ICUCAP const: {}".format(end_ICUCAP_const_time - end_HCAP_const_time))

            M.addConstrs(B_H[ti, group.name] <= group.parameters['mu'] * (group.parameters['p_H'] * obtain_I(M_test, A_test, old_forecasting, group, ti, I_vector) + obtain_Iss(M_test, A_test, old_forecasting, group, ti) * (group.parameters['p_H']
            /((group.parameters['p_ICU'] + group.parameters['p_H']) if (group.parameters['p_ICU'] + group.parameters['p_H']) != 0 else 10e-6)))
            for ti in remaining_time_steps for name,group in dynModelC.groups.items())

            end_BH_const_time = time.time()
            # print("Time spend getting BH const: {}".format(end_BH_const_time - end_ICUCAP_const_time))

            M.addConstrs(B_ICU[ti, group.name] <= group.parameters['mu'] * (group.parameters['p_ICU'] * obtain_I(M_test, A_test, old_forecasting, group, ti, I_vector) + obtain_Iss(M_test, A_test, old_forecasting, group, ti) * (group.parameters['p_ICU']
            /((group.parameters['p_ICU'] + group.parameters['p_H']) if (group.parameters['p_ICU'] + group.parameters['p_H']) != 0 else 10e-6)))
            for ti in remaining_time_steps for name, group in dynModelC.groups.items())

            end_BICU_const_time = time.time()
            # print("Time spend getting BICU const: {}".format(end_BICU_const_time - end_BH_const_time))

            M.addConstrs(M_test.sum(ti,'*') <= max_m_tests[ti] for ti in remaining_time_steps)

            end_MTEST_const_time = time.time()
            # print("Time spend getting MTEST const: {}".format(end_MTEST_const_time - end_BICU_const_time))

            M.addConstrs(A_test.sum(ti,'*') <= max_a_tests[ti] for ti in remaining_time_steps)

            end_ATEST_const_time = time.time()
            # print("Time spend getting ATEST const: {}".format(end_ATEST_const_time - end_MTEST_const_time))

            M.update()

            end_const_time = time.time()

            # print("Time spent updating the model: {}".format(end_const_time - end_ATEST_const_time))

            # print("Time building const: {}".format(end_const_time - start_const_time))

            M.setParam( 'OutputFlag', False )
            M.optimize()

            a_tests = {}
            m_tests = {}

            for name in dynModelC.groups:
                a_tests[name] = [A_test[ti, name].x for ti in remaining_time_steps]

            for name in dynModelC.groups:
                m_tests[name] = [M_test[ti, name].x for ti in remaining_time_steps]



            dynModelC.__init__(dynModelC.parameters, dynModelC.initialization,  dynModelC.dt, len(remaining_time_steps), mixing_method)

            initialize_with_forecast(dynModelC, old_forecasting)
            print(dynModelC.time_steps)

            dynModelC.simulate(change_order(m_tests), change_order(a_tests), alphas[t:])


            assign_forecastings(dynModelC, new_forecasting)

            diff = calculate_diff(new_forecasting, old_forecasting)

            print(diff)

            old_forecasting = new_forecasting

            iterations += 1
            print(iterations)
            if iterations >= max_iterations or diff <= tolerance:
                break

        # print("Time step {} took a total of {}".format(t, time.time()- time_step_time))


        for g in dynModelC.groups:
            final_m_testing[g].append(M_test[0,g].x)
            final_a_testing[g].append(A_test[0,g].x)

        for name in dynModelC.groups:
            for s in old_forecasting[name]:
                old_forecasting[name][s] = old_forecasting[name][s][1:]


    end_time = time.time()
    print("Total time to run the forecasting heuristic for {} days is: {}".format(dynModel.time_steps, end_time - start_time))
    return (change_order(final_a_testing), change_order(final_m_testing))
        #Reeinitialize the dynModel to first values of the old forecast

        #Run the simulation of the dyn model with the new test sequence and obtain the new forecast

        #Compute vector of diff





        #old is new

        #Iterations ++
        #Compare old and new forecast break if the sum of squared diff is small enough or iterations have been met print which has happened

        # append the t-th test values for a and m

def obtain_I(M_test, A_test, old_forecasting, group2, t, I_vector):
    if I_vector[group2.name][t] == float('-inf'):
        if t == 0:
            I_vector[group2.name][t] = old_forecasting[group2.name]['I'][0]

        else:
            I_vector[group2.name][t] = (I_vector[group2.name][t-1] * (1- group2.parameters['mu']) +  group2.parameters['sigma'] * old_forecasting[group2.name]['E'][t-1] - old_forecasting[group2.name]['I'][t-1] * M_test[t-1, group2.name]
            / old_forecasting[group2.name]['N'][t-1]if old_forecasting[group2.name]['N'][t-1]!=0 else 10e-6)

            # I_vector[group2.name][t] = ((1-group2.parameters['mu'])**(t-1) * (old_forecasting[group2.name]['I'][0] * (1- group2.parameters['mu']) +
            # quicksum((1-group2.parameters['mu'])**(-k) * (group2.parameters['sigma'] * old_forecasting[group2.name]['E'][k] - old_forecasting[group2.name]['I'][k] * M_test[k, group2.name]
            # / old_forecasting[group2.name]['N'][k]if old_forecasting[group2.name]['N'][k]!=0 else 10e-6)  for k in range(t))))

    return I_vector[group2.name][t]


    # return ((1-group2.parameters['mu'])**(t-1) * (old_forecasting[group2.name]['I'][0] * (1- group2.parameters['mu']) +
    # quicksum((1-group2.parameters['mu'])**(-k) * (group2.parameters['sigma'] * old_forecasting[group2.name]['E'][k] - old_forecasting[group2.name]['I'][k] * M_test[k, group2.name]
    # / old_forecasting[group2.name]['N'][k]if old_forecasting[group2.name]['N'][k]!=0 else 10e-6)  for k in range(t))))

def obtain_Infected_contact_prob(dynModelC, M_test, A_test, old_forecasting, group, t, alphas, I_vector, mixing_method):
    return quicksum(obtain_I(M_test, A_test, old_forecasting, group2, t, I_vector)
    *  n_contacts(group, group2, alphas[t], mixing_method)
    / ((old_forecasting[group2.name]['N'][t] + old_forecasting[group2.name]['Rq'][t]) if (old_forecasting[group2.name]['N'][t] + old_forecasting[group2.name]['Rq'][t])!=0 else 10e-6)
    for name2, group2 in dynModelC.groups.items())



def obtain_E(dynModelC, M_test, A_test, old_forecasting, group, t, alphas, I_vector, mixing_method, infected_contact_prob_vec):
    return (old_forecasting[group.name]['E'][t-1] * (1 - group.parameters['sigma']) + group.parameters['beta'] * old_forecasting[group.name]['S'][t-1]
    * infected_contact_prob_vec[t-1]) if t > 0 else old_forecasting[group.name]['E'][0]

def obtain_S(dynModelC, M_test, A_test, old_forecasting, group, t, alphas, I_vector, mixing_method, infected_contact_prob_vec):
    return (old_forecasting[group.name]['S'][t-1] * (1- group.parameters['beta'] * infected_contact_prob_vec[t-1])) if t>0 else old_forecasting[group.name]['S'][0]

def obtain_R(M_test, A_test, old_forecasting, group, t, I_vector):
    return (
            old_forecasting[group.name]['R'][t-1]
            + group.parameters['mu'] * (1-group.parameters['p_H'] - group.parameters['p_ICU'])
                * obtain_I(M_test, A_test, old_forecasting, group, t-1, I_vector)
            - M_test[t-1, group.name] * old_forecasting[group.name]['I'][t-1]/(old_forecasting[group.name]['N'][t-1] if old_forecasting[group.name]['N'][t-1]!=0 else 10e-6)
            ) if t>0 else old_forecasting[group.name]['R'][0]

def obtain_Ia(M_test, A_test, old_forecasting, group, t):
    return ( old_forecasting[group.name]['Ia'][t-1] * (1- group.parameters['mu']) +
            group.parameters['p_Ia'] * M_test[t-1, group.name] * (old_forecasting[group.name]['I'][t-1]/old_forecasting[group.name]['N'][t-1]if old_forecasting[group.name]['N'][t-1]!=0 else 10e-6)
    ) if t>0 else old_forecasting[group.name]['Ia'][0]

def obtain_Ips(M_test, A_test, old_forecasting, group, t):
    return ( old_forecasting[group.name]['Ips'][t-1] * (1- group.parameters['mu']) +
            group.parameters['p_Ips'] * M_test[t-1, group.name] * (old_forecasting[group.name]['I'][t-1]/old_forecasting[group.name]['N'][t-1]if old_forecasting[group.name]['N'][t-1]!=0 else 10e-6)
    )if t>0 else old_forecasting[group.name]['Ips'][0]

def obtain_Ims(M_test, A_test, old_forecasting, group, t):
    return ( old_forecasting[group.name]['Ims'][t-1] * (1- group.parameters['mu']) +
            group.parameters['p_Ims'] * M_test[t-1, group.name] * (old_forecasting[group.name]['I'][t-1]/old_forecasting[group.name]['N'][t-1]if old_forecasting[group.name]['N'][t-1]!=0 else 10e-6)
    ) if t>0 else old_forecasting[group.name]['Ims'][0]

def obtain_Iss(M_test, A_test, old_forecasting, group, t):
    return ( old_forecasting[group.name]['Iss'][t-1] * (1- group.parameters['mu']) +
            group.parameters['p_Iss'] * M_test[t-1, group.name] * (old_forecasting[group.name]['I'][t-1]/old_forecasting[group.name]['N'][t-1]if old_forecasting[group.name]['N'][t-1]!=0 else 10e-6)
    ) if t>0 else old_forecasting[group.name]['Iss'][0]


def obtain_Rq(M_test, A_test, old_forecasting, group, t):
    return( old_forecasting[group.name]['Rq'][t-1] +
            group.parameters['mu'] * (obtain_Ia(M_test, A_test, old_forecasting, group, t-1)
            + obtain_Ips(M_test, A_test, old_forecasting, group, t-1)
            + obtain_Ims(M_test, A_test, old_forecasting, group, t-1))
            + group.parameters['lambda_H_R'] * old_forecasting[group.name]['H'][t-1]
            + group.parameters['lambda_ICU_R'] * old_forecasting[group.name]['ICU'][t-1]
            + A_test[t-1, group.name] * (old_forecasting[group.name]['R'][t-1]/old_forecasting[group.name]['N'][t-1]if old_forecasting[group.name]['N'][t-1]!=0.0 else 10e-6)
    ) if t>0 else old_forecasting[group.name]['Rq'][0]

def obtain_Deaths(M_test, A_test, B_H, B_ICU,  old_forecasting, group, remaining_time_steps, I_vector):
    return (quicksum(
            B_H[t, group.name] + B_ICU[t, group.name] + group.parameters['lambda_ICU_D'] * get_ICU(M_test, A_test, B_H, B_ICU,  old_forecasting, group, t, I_vector) + group.parameters['lambda_H_D'] * get_H(M_test, A_test, B_H, B_ICU,  old_forecasting, group, t, I_vector) for t in remaining_time_steps)
    )

def get_ICU(M_test, A_test, B_H, B_ICU,  old_forecasting, group, t, I_vector):
    return (
    old_forecasting[group.name]['ICU'][t-1] * (1 - group.parameters['lambda_ICU_R'] - group.parameters['lambda_ICU_D']) - B_ICU[t-1, group.name] + group.parameters['mu'] * (group.parameters['p_ICU'] * obtain_I(M_test, A_test, old_forecasting, group, t-1, I_vector)
    + obtain_Iss(M_test, A_test, old_forecasting, group, t-1) * (group.parameters['p_ICU'] / ((group.parameters['p_ICU'] + group.parameters['p_H']) if (group.parameters['p_ICU'] + group.parameters['p_H']) != 0 else 10e-6)))
    ) if t>0 else old_forecasting[group.name]["ICU"][0]

def get_H(M_test, A_test, B_H, B_ICU, old_forecasting, group, t, I_vector):
    return (
    old_forecasting[group.name]['H'][t-1] * (1 - group.parameters['lambda_H_R'] - group.parameters['lambda_H_D']) - B_H[t-1, group.name] + group.parameters['mu'] * (group.parameters['p_H'] * obtain_I(M_test, A_test, old_forecasting, group, t-1, I_vector)
    + obtain_Iss(M_test, A_test, old_forecasting, group, t-1) * (group.parameters['p_H'] / ((group.parameters['p_ICU'] + group.parameters['p_H']) if (group.parameters['p_ICU'] + group.parameters['p_H']) != 0 else 10e-6)))
    ) if t>0 else old_forecasting[group.name]['H'][0]

def calculate_diff(new_forecasting, old_forecasting):
    total_diff = 0
    total_sums = 0
    for n in new_forecasting:
        for s in new_forecasting[n]:
            for t in range(len(new_forecasting[n][s])):
                total_diff += abs(new_forecasting[n][s][t]- old_forecasting[n][s][t])
                total_sums += 1

    return float(total_diff)

def initializeForecastings(old_forecasting, new_forecasting, dynModelC):
    for name, group in dynModelC.groups.items():
        old_forecasting[group.name] = {}
        new_forecasting[group.name] = {}

        old_forecasting[group.name]["S"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["S"][0] = float(group.S[0])

        new_forecasting[group.name]["S"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["S"][0] = float(group.S[0])

        old_forecasting[group.name]["E"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["E"][0] = float(group.E[0])

        new_forecasting[group.name]["E"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["E"][0] = float(group.E[0])

        old_forecasting[group.name]["I"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["I"][0] = float(group.I[0])

        new_forecasting[group.name]["I"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["I"][0] = float(group.I[0])

        old_forecasting[group.name]["R"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["R"][0] = float(group.R[0])

        new_forecasting[group.name]["R"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["R"][0] = float(group.R[0])

        old_forecasting[group.name]["N"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["N"][0] = float(group.N[0])

        new_forecasting[group.name]["N"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["N"][0] = float(group.N[0])

        old_forecasting[group.name]["Ia"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["Ia"][0] = float(group.Ia[0])

        new_forecasting[group.name]["Ia"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["Ia"][0] = float(group.Ia[0])

        old_forecasting[group.name]["Ips"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["Ips"][0] = float(group.Ips[0])

        new_forecasting[group.name]["Ips"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["Ips"][0] = float(group.Ips[0])

        old_forecasting[group.name]["Ims"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["Ims"][0] = float(group.Ims[0])

        new_forecasting[group.name]["Ims"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["Ims"][0] = float(group.Ims[0])

        old_forecasting[group.name]["Iss"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["Iss"][0] = float(group.Iss[0])

        new_forecasting[group.name]["Iss"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["Iss"][0] = float(group.Iss[0])

        old_forecasting[group.name]["Rq"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["Rq"][0] = float(group.Rq[0])

        new_forecasting[group.name]["Rq"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["Rq"][0] = float(group.Rq[0])

        old_forecasting[group.name]["H"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["H"][0] = float(group.H[0])

        new_forecasting[group.name]["H"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["H"][0] = float(group.H[0])

        old_forecasting[group.name]["ICU"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["ICU"][0] = float(group.ICU[0])

        new_forecasting[group.name]["ICU"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["ICU"][0] = float(group.ICU[0])

        old_forecasting[group.name]["D"] = [0 for t in range(dynModelC.time_steps)]
        old_forecasting[group.name]["D"][0] = float(group.D[0])

        new_forecasting[group.name]["D"] = [0 for t in range(dynModelC.time_steps)]
        new_forecasting[group.name]["D"][0] = float(group.D[0])


def initialize_with_forecast(dynModelC, forecasting):

    for name, group in dynModelC.groups.items():
        group.S = [float(forecasting[group.name]["S"][0])]

        group.E = [float(forecasting[group.name]["E"][0])]

        group.I = [float(forecasting[group.name]["I"][0])]

        group.R = [float(forecasting[group.name]["R"][0])]

        group.N = [float(forecasting[group.name]["N"][0])]

        group.Ia = [float(forecasting[group.name]["Ia"][0])]

        group.Ips = [float(forecasting[group.name]["Ips"][0])]

        group.Ims = [float(forecasting[group.name]["Ims"][0])]

        group.Iss = [float(forecasting[group.name]["Iss"][0])]

        group.Rq = [float(forecasting[group.name]["Rq"][0])]

        group.H = [float(forecasting[group.name]["H"][0])]

        group.ICU = [float(forecasting[group.name]["ICU"][0])]

        group.D = [float(forecasting[group.name]["D"][0])]


def assign_forecastings(dynModelC, forecasting):

    # forecasting = {}

    for name, group in dynModelC.groups.items():
        forecasting[group.name] = {}

        forecasting[group.name]["S"] = group.S

        forecasting[group.name]["E"] = group.E

        forecasting[group.name]["I"] = group.I


        forecasting[group.name]["R"] = group.R

        forecasting[group.name]["N"] = group.N

        forecasting[group.name]["Ia"] = group.Ia

        forecasting[group.name]["Ips"] =  group.Ips

        forecasting[group.name]["Ims"] = group.Ims

        forecasting[group.name]["Iss"]  = group.Iss

        forecasting[group.name]["Rq"] = group.Rq

        forecasting[group.name]["H"] = group.H

        forecasting[group.name]["ICU"]  = group.ICU

        forecasting[group.name]["D"] = group.D
