

from gurobipy import *

def forecasting_heuristic(dynModel, max_a_tests, max_m_tests, h_cap_vec, icu_cap_vec, tolerance, max_iterations):
    #Create copy of dyn model to modify
    dynModelC = DynamicalModel(dynModel.parameters, dynModel.dt, dynModel.time_steps)

    #Initialize real testing vectors
    final_a_testing = {}
    final_m_testing = {}
    for g in dynModelC.groups:
        final_a_testing[g] = []
        final_m_testing[g] = []

    #Initialize old and new forecasting to zeros except for the first elements
    old_forecasting[group.name] = {}
    new_forecasting[group.name] = {}
    for group in dynModelC.groupss:
    old_forecasting[group.name][group.name] = {}
    new_forecasting[group.name][group.name] = {}

    old_forecasting[group.name]["S"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["S"][0] = group.S[0]

    new_forecasting[group.name]["S"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["S"][0] = group.S[0]

    old_forecasting[group.name]["E"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["E"][0] = group.E[0]

    new_forecasting[group.name]["E"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["E"][0] = group.E[0]

    old_forecasting[group.name]["I"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["I"][0] = group.I[0]

    new_forecasting[group.name]["I"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["I"][0] = group.I[0]

    old_forecasting[group.name]["R"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["R"][0] = group.R[0]

    new_forecasting[group.name]["R"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["R"][0] = group.R[0]

    old_forecasting[group.name]["N"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["N"][0] = group.N[0]

    new_forecasting[group.name]["N"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["N"][0] = group.N[0]

    old_forecasting[group.name]["Ia"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["Ia"][0] = group.Ia[0]

    new_forecasting[group.name]["Ia"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["Ia"][0] = group.Ia[0]

    old_forecasting[group.name]["Ips"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["Ips"][0] = group.Ips[0]

    new_forecasting[group.name]["Ips"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["Ips"][0] = group.Ips[0]

    old_forecasting[group.name]["Ims"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["Ims"][0] = group.Ims[0]

    new_forecasting[group.name]["Ims"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["Ims"][0] = group.Ims[0]

    old_forecasting[group.name]["Iss"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["Iss"][0] = group.Iss[0]

    new_forecasting[group.name]["Iss"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["Iss"][0] = group.Iss[0]

    old_forecasting[group.name]["Rq"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["Rq"][0] = group.Rq[0]

    new_forecasting[group.name]["Rq"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["Rq"][0] = group.Rq[0]

    old_forecasting[group.name]["H"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["H"][0] = group.H[0]

    new_forecasting[group.name]["H"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["H"][0] = group.H[0]

    old_forecasting[group.name]["ICU"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["ICU"][0] = group.ICU[0]

    new_forecasting[group.name]["ICU"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["ICU"][0] = group.ICU[0]

    old_forecasting[group.name]["D"] = [0 for t in dynModelC.time_steps]
    old_forecasting[group.name]["D"][0] = group.D[0]

    new_forecasting[group.name]["D"] = [0 for t in dynModelC.time_steps]
    new_forecasting[group.name]["D"][0] = group.D[0]



    #For all times time_steps
    for t in dynModel.time_steps:
        # Set the correct time steps
        time_steps = range(len(dynModel.time_steps) - t)

        # Empty the copy of the dyn model and put as initial conditions the first elements of the new (old?) forecast.
        dynModelC.__init__(dynModelC.parameters, dynModelC.dt, time_steps)
        for group in dynModelC.groups:
        group.S = [old_forecasting[group.name]["S"][0]]

        group.E = [old_forecasting[group.name]["E"][0]]

        group.I = [old_forecasting[group.name]["I"][0]]

        group.R = [old_forecasting[group.name]["R"][0]]

        group.N = [old_forecasting[group.name]["N"][0]]

        group.Ia = [old_forecasting[group.name]["Ia"][0]]

        group.Ips = [old_forecasting[group.name]["Ips"][0]]

        group.Ims = [old_forecasting[group.name]["Ims"][0]]

        group.Iss = [old_forecasting[group.name]["Iss"][0]]

        group.Rq = [old_forecasting[group.name]["Rq"][0]]

        group.H = [old_forecasting[group.name]["H"][0]]

        group.ICU = [old_forecasting[group.name]["ICU"][0]]

        group.D = [old_forecasting[group.name]["D"][0]]


        # Run the model with no testing and fix the results as the old forecasting
        # Fix the new forecast as all all zeros
        no_m_tests = [0 for i in range time_steps]
        no_a_tests = [0 for i in range time_steps]

        dynModelC.simulate(no_m_tests, no_a_tests, h_cap_vec[t:], icu_cap_vec[t:])

        old_forecasting = {}
        new_forecasting = {}
        for group in dynModelC.groups:
        old_forecasting[group.name] = {}
        new_forecasting[group.name] = {}

        old_forecasting[group.name]["S"] = group.S

        new_forecasting[group.name]["S"] = group.S

        old_forecasting[group.name]["E"] = group.E

        new_forecasting[group.name]["E"] = group.E

        old_forecasting[group.name]["I"] = group.I

        new_forecasting[group.name]["I"] = group.I

        old_forecasting[group.name]["R"] = group.R

        new_forecasting[group.name]["R"] = group.R

        old_forecasting[group.name]["N"] = group.N

        new_forecasting[group.name]["N"] = group.N

        old_forecasting[group.name]["Ia"] = group.Ia

        new_forecasting[group.name]["Ia"] = group.Ia

        old_forecasting[group.name]["Ips"] =  group.Ips

        new_forecasting[group.name]["Ips"] =  group.Ips

        old_forecasting[group.name]["Ims"] = group.Ims

        new_forecasting[group.name]["Ims"]  = group.Ims

        old_forecasting[group.name]["Iss"]  = group.Iss

        new_forecasting[group.name]["Iss"]  = group.Iss

        old_forecasting[group.name]["Rq"] = group.Rq

        new_forecasting[group.name]["Rq"] = group.Rq

        old_forecasting[group.name]["H"] = group.H

        new_forecasting[group.name]["H"] = group.H

        old_forecasting[group.name]["ICU"]  = group.ICU

        new_forecasting[group.name]["ICU"]  = group.ICU

        old_forecasting[group.name]["D"] = group.D

        new_forecasting[group.name]["D"] = group.D



        # iterations = 0
        #While true do (will break only when the number of iterations have completed or the tolerance level has been reached)

        #Write gurobi problem with fixed states to be the old forecast and obtain a seq of m and a tests

        #Reeinitialize the dynModel to first values of the old forecast

        #Run the simulation of the dyn model with the new test sequence and obtain the new forecast

        #Compute vector of diff

        #old is new

        #Iterations ++
        #Compare old and new forecast break if the sum of squared diff is small enough or iterations have been met print which has happened

        # append the t-th test values for a and m

        # iterations = 0
        iterations = 0

        #Create G Model (so as to modify it later)
        M = Model()
        B_ICU = M.addVars(time_steps, dynModelC.groupss, vtype=GRB.CONTINUOUS, name="ICU-Bounces")
        B_H = M.addVars(time_steps,dynModelC.groupss, vtype=GRB.CONTINUOUS, name="H-Bounces")
        A_test = M.addVars(time_steps, dynModelC.groupss, vtype=GRB.CONTINUOUS, ub=max_a_tests, name="a-tests")
        M_test = M.addVars(time_steps, dynModelC.groupss, vtype=GRB.CONTINUOUS, ub=max_m_tests, name="m-tests")



        #While true do (will break only when the number of iterations have completed or the tolerance level has been reached)
        while True:



            #Write gurobi problem with fixed states to be the old forecast and obtain a seq of m and a tests
            #Objective

            economic_obj = quicksum(group.parameters['v_unconf'] * (obtain_E(M_test, A_test, old_forecasting, group, t) + obtain_S(M_test, A_test, old_forecasting, group, t) + obtain_R(M_test, A_test, old_forecasting, group, t)) + group.parameters['v_conf'] * obtain_Rq(M_test, A_test, old_forecasting, group, t) for t in time_steps for group in dynModelC.groupss)

            deaths = quicksum(group.parameters['v_deaths'] * obtain_Deaths(M_test, A_test, B_H, B_ICU, old_forecasting, group, time_steps) for group in dynModelC.groupss)

            obj = economic_obj - deaths

            M.setObjective(obj, GRB.MAXIMIZE)

            M.addConstrs((quicksum(group.paramters['mu'] * (group.paramters['p_H'] * obtain_I(M_test, A_test, old_forecasting, group, t)) + (group.paramters['p_H']/(group.paramters['p_H'] + group.paramters['p_ICU']))
            * obtain_Iss(M_test,A_test,old_forecasting,goup,t) - B_H[t, group] for group in dynModelC.groupss) <= h_cap_vec[t] - quicksum((1 - group2.parameters['lambda_H_R'] - group2.parameters['lambda_H_D']) get_H(M_test, A_test, B_H, B_ICU,  old_forecasting, group2, t) for group2 in dynModelC.groupss) for t in time_steps), name="HCAP")

            M.addConstrs((quicksum(group.paramters['mu'] * (group.paramters['p_ICU'] * obtain_I(M_test, A_test, old_forecasting, group, t)) + (group.paramters['p_ICU']/(group.paramters['p_H'] + group.paramters['p_ICU']))
            * obtain_Iss(M_test,A_test,old_forecasting,goup,t) - B_H[t, group] for group in dynModelC.groupss) <= h_cap_vec[t] - quicksum((1 - group2.parameters['lambda_ICU_R'] - group2.parameters['lambda_ICU_D']) get_ICU(M_test, A_test, B_H, B_ICU,  old_forecasting, group2, t) for group2 in dynModelC.groupss) for t in time_steps), name="ICUCAP")

            M.addConstrs(B_H[t, group] <= group.paramters['mu'] * (group.paramters['p_H'] * obtain_I(M_test, A_test, old_forecasting, group, t) + obtain_Iss(M_test, A_test, old_forecasting, group, t) * (group.parameters['p_H']/(group.parameters['p_H'] + group.parameters['p_ICU'])))
            for t in time_steps for group in dynModelC.groups)

            M.addConstrs(B_ICU[t, group] <= group.paramters['mu'] * (group.paramters['p_ICU'] * obtain_I(M_test, A_test, old_forecasting, group, t) + obtain_Iss(M_test, A_test, old_forecasting, group, t) * (group.parameters['p_ICU']/(group.parameters['p_H'] + group.parameters['p_ICU'])))
            for t in time_steps for group in dynModelC.groups)

            M.addConst(quicksum(M_test[t,group] for group in dynModelC.groups) <= max_m_tests)


            M.addConst(quicksum(A_test[t,group] for group in dynModelC.groups) <= max_a_tests)

            M.update()

            M.optimize()

            a_tests = {}
            m_tests = {}

            for name in dynModelC.groups:
                a_tests[name] = [A_test[t, name].x for t in time_steps]

                for name in dynModelC.groups:
                m_tests[name] = [M_test[t, name].x for t in time_steps]

            dynModelC.__init__(dynModelC.parameters, dynModelC.dt, time_steps)

            for group in dynModelC.groups:
                group.S = [old_forecasting[group.name]["S"][0]]

                group.E = [old_forecasting[group.name]["E"][0]]

                group.I = [old_forecasting[group.name]["I"][0]]

                group.R = [old_forecasting[group.name]["R"][0]]

                group.N = [old_forecasting[group.name]["N"][0]]

                group.Ia = [old_forecasting[group.name]["Ia"][0]]

                group.Ips = [old_forecasting[group.name]["Ips"][0]]

                group.Ims = [old_forecasting[group.name]["Ims"][0]]

                group.Iss = [old_forecasting[group.name]["Iss"][0]]

                group.Rq = [old_forecasting[group.name]["Rq"][0]]

                group.H = [old_forecasting[group.name]["H"][0]]

                group.ICU = [old_forecasting[group.name]["ICU"][0]]

                group.D = [old_forecasting[group.name]["D"][0]]

            dynModelC.simulate(m_tests, a_tests, h_cap_vec[t:], icu_cap_vec[t:])



            for group in dynModelC.groups:
                new_forecasting[group.name]["S"] = group.S

                new_forecasting[group.name]["E"] = group.E

                new_forecasting[group.name]["I"] = group.I

                new_forecasting[group.name]["R"] = group.R

                new_forecasting[group.name]["N"] = group.N


                new_forecasting[group.name]["Ia"] = group.Ia


                new_forecasting[group.name]["Ips"] =  group.Ips


                new_forecasting[group.name]["Ims"]  = group.Ims


                new_forecasting[group.name]["Iss"]  = group.Iss


                new_forecasting[group.name]["Rq"] = group.Rq

                new_forecasting[group.name]["H"] = group.H


                new_forecasting[group.name]["ICU"]  = group.ICU

                new_forecasting[group.name]["D"] = group.D



        # iterations = 0
        #While true do (will break only when the number of iterations have completed or the tolerance level has been reached)

          #Write gurobi problem with fixed states to be the old forecast and obtain a seq of m and a tests

          #Reeinitialize the dynModel to first values of the old forecast

          #Run the simulation of the dyn model with the new test sequence and obtain the new forecast

          #Compute vector of diff

          #old is new

          #Iterations ++
          #Compare old and new forecast break if the sum of squared diff is small enough or iterations have been met print which has happened

        # append the t-th test values for a and m

            diff = calculate_diff(new_forecasting, old_forecasting)

            old_forecasting = new_forecasting

            iterations += 1

            if iterations >= max_iterations or diff <= tolerance:
                break



        for g in dynModelC.groups:
            final_m_testing[g].append(M_test[0,g].x)
            final_a_testing[g].append(A_test[0,g].x)

    return (final_a_testing, final_m_testing)
        #Reeinitialize the dynModel to first values of the old forecast

        #Run the simulation of the dyn model with the new test sequence and obtain the new forecast

        #Compute vector of diff





        #old is new

        #Iterations ++
        #Compare old and new forecast break if the sum of squared diff is small enough or iterations have been met print which has happened

        # append the t-th test values for a and m

def obtain_I(M_test, A_test, old_forecasting, group2, t):
    return ((1-group2.parameters['mu'])**(t-1) * (old_forecasting[group2.name]['I'][0] * (1- group2.parameters['mu']) +
    quicksum([(1-group2.paramters['mu'])**(-k) * (group2.paramters['sigma'] * old_forecasting[group2.name]['E'][k] - old_forecasting[group2.name][I][k] * M_test[k, group2] / old_forecasting[group2.name]['N'][k])  for k in range(t)])))


def obtain_E(M_test, A_test, old_forecasting, group, t):
    return (old_forecasting[group.name]['E'][t] * (1 - group.paramters['sigma']) + group.paramters['beta'] * old_forecasting[group.name]['S'][t]
    * quicksum(obtain_I(M_test, A_test, old_forecasting, group2, t)
    *  group.contacts[group2] / (sum([old_forecasting[group3.name]['N'][t] + old_forecasting[group3.name]['Rq'][t] for group3 in group2.same_biomarkers])) for group2 in group.all_groups))

def obtain_S(M_test, A_test, old_forecasting, group, t):
    return (old_forecasting[group.name]['S'][t] * (1- group.paramters['beta'] * quicksum(obtain_I(M_test, A_test, old_forecasting, group2, t)
    *  group.contacts[group2] / (sum([old_forecasting[group3.name]['N'][t] + old_forecasting[group3.name]['Rq'][t] for group3 in group2.same_biomarkers])) for group2 in group.all_groups)))

def obtain_R(M_test, A_test, old_forecasting, group, t):
    return (
            old_forecasting[group.name]['R'][t]
            + group.parameters['mu'] * (1-group.paramters['p_H'] - group.paramters['p_ICU'])
                * obtain_I(M_test, A_test, old_forecasting, group, t)
            - M_test[t, group] * (old_forecasting[group.name]['I'][t]/old_forecasting[group.name]['N'][t])
            )

def obtain_Ia(M_test, A_test, old_forecasting, group, t):
    return ( old_forecasting[group.name]['Ia'][t] * (1- group.parameters['mu']) +
            group.parameters['p_Ia'] * M_test[t, group] * (old_forecasting[group.name]['I'][t]/old_forecasting[group.name]['N'][t])
    )

def obtain_Ips(M_test, A_test, old_forecasting, group, t):
    return ( old_forecasting[group.name]['Ips'][t] * (1- group.parameters['mu']) +
            group.parameters['p_Ips'] * M_test[t, group] * (old_forecasting[group.name]['I'][t]/old_forecasting[group.name]['N'][t])
    )

def obtain_Ims(M_test, A_test, old_forecasting, group, t):
    return ( old_forecasting[group.name]['Ims'][t] * (1- group.parameters['mu']) +
            group.parameters['p_Ims'] * M_test[t, group] * (old_forecasting[group.name]['I'][t]/old_forecasting[group.name]['N'][t])
    )
def obtain_Iss(M_test, A_test, old_forecasting, group, t):
    return ( old_forecasting[group.name]['Iss'][t] * (1- group.parameters['mu']) +
            group.parameters['p_Iss'] * M_test[t, group] * (old_forecasting[group.name]['I'][t]/old_forecasting[group.name]['N'][t])
    )


def obtain_Rq(M_test, A_test, old_forecasting, group, t):
    return( old_forecasting[group.name]['Rq'][t] +
            group.parameters['mu'] * (obtain_Ia(M_test, A_test, old_forecasting, group, t)
            + obtain_Ips(M_test, A_test, old_forecasting, group, t)
            + obtain_Ims(M_test, A_test, old_forecasting, group, t))
            + group.paramters['lambda_H_R'] * old_forecasting[group.name]['H'][t]
            + group.paramters['lambda_ICU_R'] * old_forecasting[group.name]['ICU'][t]
            + A_test[t, group] * (old_forecasting[group.name]['R'][t]/old_forecasting[group.name]['N'][t])
    )

def obtain_Deaths(M_test, A_test, B_H, B_ICU,  old_forecasting, group, time_steps):
    return (quicksum(
            B_H[t, group] + B_ICU[t, group] + group.paramters['lambda_ICU_D'] * get_ICU(M_test, A_test, B_H, B_ICU,  old_forecasting, group, t) + group.paramters['lambda_H_D'] * get_H(M_test, A_test, B_H, B_ICU,  old_forecasting, group, t) for t in time_steps)
    )

def get_ICU(M_test, A_test, B_H, B_ICU,  old_forecasting, group, t):
    return (
    old_forecasting[group.name]['ICU'][t] * (1 - group.paramters['lambda_ICU_R'] - group.paramters['lambda_ICU_D']) - B_ICU[t, group] + group.paramters['mu'] * (group.parameters['p_ICU'] * obtain_I(M_test, A_test, old_forecasting, group2, t)
    + obtain_Iss(M_test, A_test, old_forecasting, group, t) * (group.parameters['p_ICU'] / (group.parameters['p_ICU'] + group.parameters['p_H'])))
    )

def get_H(M_test, A_test, B_H, B_ICU,  old_forecasting, group, t):
    return (
    old_forecasting[group.name]['H'][t] * (1 - group.paramters['lambda_H_R'] - group.paramters['lambda_H_D']) - B_H[t, group] + group.paramters['mu'] * (group.parameters['p_H'] * obtain_I(M_test, A_test, old_forecasting, group2, t)
    + obtain_Iss(M_test, A_test, old_forecasting, group, t) * (group.parameters['p_H'] / (group.parameters['p_H'] + group.parameters['p_ICU'])))
    )

def calculate_diff(new_forecasting, old_forecasting):
    diff = 0
    for n in new_forecasting:
        for s in new_forecasting[n]:
            for t in range(len(new_forecasting[n][s])):
                diff += abs(new_forecasting[n][s][t]- old_forecasting[n][s][t])

    return diff
