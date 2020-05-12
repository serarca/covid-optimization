import random
import numpy as np
from collections import defaultdict
from gurobipy import *

# A heuristic that assigns all testing to a given group
def all_to_one(dyn_model, max_a_tests, max_m_tests):
	# Choose a group randomly
	group = random.choice(dyn_model.groups.keys())
	print("Chose group '%s' to give all testing"%group)
	a_tests = {}
	m_tests = {}
	for name in dyn_model.groups:
		if name == group:
			a_tests[name] = max_a_tests[:]
			m_tests[name] = max_m_tests[:]
		else:
			a_tests[name] = [0 for t in range(dyn_model.time_steps)]
			m_tests[name] = [0 for t in range(dyn_model.time_steps)]

	return (a_tests,m_tests)


# A heuristic that assigns random testing at each point in time
def random_partition(dyn_model, max_a_tests, max_m_tests):
	# Choose a group randomly
	a_sample = defaultdict(list)
	m_sample = defaultdict(list)
	# Sample dictionary of A tests for all groups at all times uniformly from the simplex boundary
	for t in range(dyn_model.time_steps):
		sample_sum = 0
		for n in dyn_model.groups:
			sample = np.random.uniform()
			a_sample[n].append(sample)
			sample_sum += sample
		for n in dyn_model.groups:
			a_sample[n][t] = a_sample[n][t]/sample_sum*max_a_tests[t]
	# Sample dictionary of M tests for all groups at all times uniformly from the simplex boundary
	for t in range(dyn_model.time_steps):
		sample_sum = 0
		for n in dyn_model.groups:
			sample = np.random.uniform()
			m_sample[n].append(sample)
			sample_sum += sample
		for n in dyn_model.groups:
			m_sample[n][t] = m_sample[n][t]/sample_sum*max_m_tests[t]

	return (a_sample,m_sample)

def forecasting_heuristic(dynModel, max_a_tests, max_m_tests, h_cap_vec, icu_cap_vec, tolerance, max_iterations):

	#Create copy of dyn model to modify
	dynModelC = DynamicalModel(dynModel.parameters, dynModel.dt, dynModel.time_steps)

	#Initialize real testing vectors
	final_a_testing = []
	final_m_testing = []

	#Initialize old and new forecasting to zeros except for the first elements
	old_forecasting[group.name] = {}
	new_forecasting[group.name] = {}
	for group in dynModelC.all_groups:
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
		time_steps = range(len(dynModel.time_steps) - 1)

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


		for group in dynModelC.all_groups:
			old_forecasting[group.name][group.name] = {}
			new_forecasting[group.name][group.name] = {}

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
		B_ICU = M.addVars(time_steps, vtype=GRB.CONTINUOUS, name="ICU-Bounces")
		B_H = M.addVars(time_steps, vtype=GRB.CONTINUOUS, name="H-Bounces")
		A_test = M.addVars(time_steps, vtype=GRB.CONTINUOUS, ub=max_a_tests, name="a-tests")
		M_test = M.addVars(time_steps, vtype=GRB.CONTINUOUS, ub=max_m_tests, name="m-tests")



		#While true do (will break only when the number of iterations have completed or the tolerance level has been reached)
		while True:

			#Write gurobi problem with fixed states to be the old forecast and obtain a seq of m and a tests
			#Objective
			obj = quicksum(group.parameters['v_unconf'] *
			 ((old_forecasting[group.name]['S'][t] - group.parameters['beta'] * old_forecasting[group.name]['S'][t] * sum([group.contacts[group2] *
			(old_forecasting[group2]['I'][t] /old_forecasting[group2]['N'][t]) for group2 in group.all_groups()])) + (old_forecasting[group.name]['E'][t] + group.parameters['beta'] * old_forecasting[group.name]['S'][t] * sum([group.contacts[group2] *
			(old_forecasting[group2]['I'][t]/old_forecasting[group2]['N'][t]) for group2 in group.all_groups()]) - group.parameters['sigma'] * old_forecasting[group.name]['E'][t]) +
			 (old_forecasting[group.name]['R'][t] + group.parameters['mu'] * (1 - group.parameters['p_H'] - group.parameters['p_ICU']) * old_forecasting[group.name]['I'][t] - M_test[t] * (old_forecasting[group.name]['I'][t] /
			 (old_forecasting[group.name]['N'][t] - old_forecasting[group.name]['R'][t]))))
			 +  group.parameters['v_conf'] *
			 (old_forecasting[group.name]['R'][t] + group.parameters['mu'] * (old_forecasting[group.name]['Ia'][t]+old_forecasting[group.name]['Ips'][t]+old_forecasting[group.name]['Ims'][t] + group.parameters['lambda_H_R'] * old_forecasting[group.name]['H'][t]
			 + group.parameters['lambda_ICU_R'] * old_forecasting[group.name]['ICU'][t]) + A_test[t] * (old_forecasting[group.name]['R'][t]/
			 (old_forecasting[group.name]['N'][t] - old_forecasting[group.name]['Rq'][t])))
			 + )
			
			#Reeinitialize the dynModel to first values of the old forecast

			#Run the simulation of the dyn model with the new test sequence and obtain the new forecast

			#Compute vector of diff

			#old is new

			#Iterations ++
			#Compare old and new forecast break if the sum of squared diff is small enough or iterations have been met print which has happened

		# append the t-th test values for a and m
