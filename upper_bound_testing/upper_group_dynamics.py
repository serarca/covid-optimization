from collections import defaultdict
from bound import Bounds
import numpy as np
import pandas as pd
import math

def n_contacts(group_g, group_h, alphas, mixing_method):
	n = 0
	if mixing_method['name'] == "maxmin":
		for activity in alphas[group_g.name]:
			n += group_g.contacts[activity][group_h.name]*(
					(alphas[group_g.name][activity]*math.exp(alphas[group_g.name][activity]*mixing_method['param']) + alphas[group_h.name][activity]*math.exp(alphas[group_h.name][activity]*mixing_method['param']))
					/(math.exp(alphas[group_g.name][activity]*mixing_method['param'])+math.exp(alphas[group_h.name][activity]*mixing_method['param']))
				)
	elif mixing_method['name'] == "mult":
		for activity in alphas[group_g.name]:
			n += group_g.contacts[activity][group_h.name]*alphas[group_g.name][activity]*alphas[group_h.name][activity]
	elif mixing_method['name'] == "min":
		for activity in alphas[group_g.name]:
			n += group_g.contacts[activity][group_h.name]*min(alphas[group_g.name][activity],alphas[group_h.name][activity])	
	elif mixing_method['name'] == "max":
		for activity in alphas[group_g.name]:
			n += group_g.contacts[activity][group_h.name]*max(alphas[group_g.name][activity],alphas[group_h.name][activity])	
	else:
		assert(False)
	
	return n


class DynamicalModelUpper:
	def __init__(self, parameters, initialization, dt, time_steps, mixing_method):
		self.parameters = parameters
		self.t = 0
		self.dt = dt
		self.time_steps = time_steps
		self.initialization = initialization
		self.mixing_method = mixing_method

		# Create groups from parameters
		self.groups = {}
		for n in parameters['seir-groups']:
			self.groups[n] = SEIR_group_upper(parameters['seir-groups'][n], initialization[n], self.dt, self.mixing_method, self.time_steps, self)

		# Attach other groups to each group
		for n in self.groups:
			self.groups[n].attach_other_groups(self.groups)

		# Fix number of beds and icus
		self.beds = self.parameters['global-parameters']['C_H']
		self.icus = self.parameters['global-parameters']['C_ICU']

		# Initialize objective values
		self.economic_values = [float("nan")]
		self.rewards = [float("nan")]
		self.deaths = [float("nan")]

		# Initialize total population
		self.total_population = sum([sum([initialization[group][cat] for cat in initialization[group].keys()]) for group in initialization.keys()])


	def take_time_step(self, m_tests, a_tests, alphas):
		for n in self.groups:
			self.groups[n].update_total_contacts(self.t, alphas)
		for n in self.groups:
			self.groups[n].take_time_step(m_tests[n], a_tests[n], self.beds, self.icus)


		# Calculate economic values
		state = self.get_state(self.t+1)
		deaths = sum([group.D[self.t+1]-group.D[self.t] for name,group in self.groups.items()])
		deaths_value = sum([(group.D[self.t+1]-group.D[self.t])*group.economics['death_value'] for name,group in self.groups.items()])
		economic_value = self.get_economic_value(state, alphas)
		reward = economic_value - deaths_value
		result = {
			"state": state,
			"economic_value": economic_value,
			"deaths": deaths,
			"deaths_value": deaths_value,
			"reward":reward,
		}

		# Update economic values
		self.economic_values.append(economic_value)
		self.deaths.append(deaths)
		self.rewards.append(reward)

		# Update time
		self.t += 1
		return result

	# Simulates the dynamics given a vector of molecular tests, atomic tests and alphas
	def simulate(self, m_tests_vec, a_tests_vec, alphas_vec):
		for t in range(self.time_steps):
			self.take_time_step(m_tests_vec[t], a_tests_vec[t], alphas_vec[t])

	# Given a state and set of alphas, returns the economic value
	def get_economic_value(self, state, alphas):
		value = 0
		for group in state:
			value = value + (
				self.groups[group].economics['work_value']*(
					alphas[group]['work']+
					self.groups[group].economics['lockdown_fraction']*(1-alphas[group]['work'])
				)*
				(state[group]["S"] + state[group]["E"] + state[group]["R"])
				* self.dt
			)
			# Liberate people in Rq group
			value = value + state[group]["Rq"]*self.groups[group].economics['work_value']* self.dt
		return value

	def get_state(self, t):
		state = {}
		for name,group in self.groups.items():
			state[name] = {
				"S": group.S[t],
				"E": group.E[t],
				"I": group.I[t],
				"R": group.R[t],
				"N": group.N[t],
				"Ia": group.Ia[t],
				"Ips": group.Ips[t],
				"Ims": group.Ims[t],
				"Iss": group.Iss[t],
				"Rq": group.Rq[t],
				"H": group.H[t],
				"ICU": group.ICU[t],
				"D": group.D[t],
			}
		return state

	def get_total_deaths(self):
		total = 0
		for t in range(1,self.time_steps+1):
			total += self.deaths[t]
		return total

	def get_total_economic_value(self):
		total = 0
		for t in range(1,self.time_steps+1):
			total += self.economic_values[t]
		return total

	def get_total_reward(self):
		total = 0
		for t in range(1,self.time_steps+1):
			total += self.rewards[t]
		return total

	def print_stats(self):
		print("Economic Value: "+str(self.get_total_economic_value()))
		print("Deaths "+str(self.get_total_deaths()))
		print("Total Reward "+str(self.get_total_reward()))


class SEIR_group_upper:
	# Time step
	def __init__(self, group_parameters, group_initialization, dt, mixing_method, time_steps, parent):
		# Group name
		self.name = group_parameters['name']
		self.parameters = group_parameters['parameters']
		self.contacts = group_parameters['contacts']
		self.economics = group_parameters['economics']
		self.initial_conditions = group_initialization
		self.mixing_method = mixing_method
		self.time_steps = time_steps
		self.parent = parent
		self.initialize_vars(self.initial_conditions)



		# Time step
		self.t = 0
		self.dt = dt


	def initialize_vars(self, initial_conditions):
		# Susceptible
		self.S = [float(initial_conditions['S'])]
		# Exposed (unquarantined)
		self.E = [float(initial_conditions['E'])]
		# Infected (unquarantined)
		self.I = [float(initial_conditions['I'])]
		# Recovered (unquarantined)
		self.R = [float(initial_conditions['R'])]
		# Unquarantined patients
		self.N = [self.S[0] + self.E[0] + self.I[0]+ self.R[0]]

		# Infected quarantined with different degrees of severity
		self.Ia = [float(initial_conditions['Ia'])]
		self.Ips = [float(initial_conditions['Ips'])]
		self.Ims = [float(initial_conditions['Ims'])]
		self.Iss = [float(initial_conditions['Iss'])]

		# Recovered quanrantined
		self.Rq = [float(initial_conditions['Rq'])]

		# In hospital bed
		self.H = [float(initial_conditions['H'])]
		# In ICU
		self.ICU = [float(initial_conditions['ICU'])]
		# Dead
		self.D = [float(initial_conditions['D'])]
		# Economic value
		self.Econ = [0.0]

		# Contacts
		self.total_contacts = []

		# The initial population
		self.N0 = self.S[0] + self.E[0] + self.I[0]+ self.R[0] + self.Rq[0]


	def update_total_contacts(self, t, alphas):
		if (len(self.total_contacts) == t):
			summ_contacts = 0
			for n,g in self.all_groups.items():
				# Set population the same as N0
				pop_g = self.N0
				new_contacts = n_contacts(self, g, alphas, self.mixing_method)
				summ_contacts += new_contacts*g.I[t]/(pop_g if pop_g!=0 else 10e-6)
			self.total_contacts.append(summ_contacts*self.S[t])
		else:
			assert(False)


	# Attach other groups to make it easier to find variables of other groups
	def attach_other_groups(self,all_groups):
		self.all_groups = all_groups

	# Advances one time step, given the m_tests and a_tests variable
	def take_time_step(self, m_tests, a_tests, h_cap, icu_cap):
		self.update_N(m_tests, a_tests)
		self.update_S(m_tests, a_tests)
		self.update_E(m_tests, a_tests)
		self.update_I(m_tests, a_tests)
		self.update_R(m_tests, a_tests)
		self.update_Ia(m_tests, a_tests)
		self.update_Ips(m_tests, a_tests)
		self.update_Ims(m_tests, a_tests)
		self.update_Iss(m_tests, a_tests)
		self.update_Rq(m_tests, a_tests)
		self.update_H(m_tests, a_tests, h_cap, icu_cap)
		self.update_ICU(m_tests, a_tests, h_cap, icu_cap)
		self.update_D(m_tests, a_tests, h_cap, icu_cap)

		self.t += 1

	# Gives flow of how many people flowing to H
	def flow_H(self, t):
		if self.parameters['p_H'] != 0.0:
			return self.parameters['mu']*self.parameters['p_H']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
		else:
			return 0.0

	# Gives flow of how many people flowing to ICU
	def flow_ICU(self, t):
		if self.parameters['p_ICU'] != 0.0:
			return self.parameters['mu']*self.parameters['p_ICU']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
		else:
			return 0.0

	# Updates N
	def update_N(self, m_tests, a_tests):
		delta_N = (
			- m_tests
			- a_tests
			- self.parameters['mu']*(self.parameters['p_H'] + self.parameters['p_ICU'])*self.I[self.t]
		)
		self.N += [self.N[self.t]+delta_N*self.dt]

	# Updates S
	def update_S(self, m_tests, a_tests):
		delta_S = -self.parameters['beta']*self.total_contacts[self.t-1]
		self.S += [self.S[self.t]+delta_S*self.dt]

	# Updates Exposed
	def update_E(self, m_tests, a_tests):
		delta_E = self.parameters['beta']*self.total_contacts[self.t-1] - self.parameters['sigma']*self.E[self.t]
		self.E += [self.E[self.t]+delta_E*self.dt]


	# Updates infected
	def update_I(self, m_tests, a_tests):
		delta_I = self.parameters['sigma']*self.E[self.t] - self.parameters['mu']*self.I[self.t] - m_tests
		self.I += [self.I[self.t]+delta_I*self.dt]


	# Updates recovered
	def update_R(self, m_tests, a_tests):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]-self.parameters["p_ICU"])*self.I[self.t] - a_tests
		self.R += [self.R[self.t]+delta_R*self.dt]


	# Updates infected in quarantine
	def update_Ia(self, m_tests, a_tests):
		delta_Ia = self.parameters['p_Ia']*m_tests - self.parameters['mu']*self.Ia[self.t]
		self.Ia += [self.Ia[self.t]+delta_Ia*self.dt]

	def update_Ips(self, m_tests, a_tests):
		delta_Ips = self.parameters['p_Ips']*m_tests - self.parameters['mu']*self.Ips[self.t]
		self.Ips += [self.Ips[self.t]+delta_Ips*self.dt]

	def update_Ims(self, m_tests, a_tests):
		delta_Ims = self.parameters['p_Ims']*m_tests - self.parameters['mu']*self.Ims[self.t]
		self.Ims += [self.Ims[self.t]+delta_Ims*self.dt]

	def update_Iss(self, m_tests, a_tests):
		delta_Iss = self.parameters['p_Iss']*m_tests - self.parameters['mu']*self.Iss[self.t]
		self.Iss += [self.Iss[self.t]+delta_Iss*self.dt]


	# Update recovered in quarentine
	def update_Rq(self, m_tests, a_tests):
		delta_Rq = (
			self.parameters['mu']*(self.Ia[self.t]+self.Ips[self.t]+self.Ims[self.t]) +
			self.parameters['lambda_H_R']*self.H[self.t] +
			self.parameters['lambda_ICU_R']*self.ICU[self.t] +
			a_tests
		)
		self.Rq += [self.Rq[self.t]+delta_Rq*self.dt]


	def update_H(self, m_tests, a_tests, h_cap, icu_cap):
		# For each group, calculate the entering amount
		entering_h = {}
		summ_entering_h = 0
		summ_staying_h = 0
		for n,g in self.all_groups.items():
			entering_h[n] = self.all_groups[n].flow_H(self.t)
			summ_entering_h += entering_h[n]
			summ_staying_h += (1-g.parameters['lambda_H_R']-g.parameters['lambda_H_D'])*g.H[self.t]

		delta_H = (
			- (self.parameters["lambda_H_R"] + self.parameters["lambda_H_D"])*self.H[self.t]
			+ entering_h[self.name]*(1-(summ_entering_h-h_cap+summ_staying_h if summ_entering_h-h_cap+summ_staying_h>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6))
		)
		self.H += [self.H[self.t]+delta_H*self.dt]


	def update_ICU(self, m_tests, a_tests, h_cap, icu_cap):
		# For each group, calculate the entering amount
		entering_icu = {}
		summ_entering_icu = 0
		summ_staying_icu = 0
		for n,g in self.all_groups.items():
			entering_icu[n] = self.all_groups[n].flow_ICU(self.t)
			summ_entering_icu += entering_icu[n]
			summ_staying_icu += (1-g.parameters['lambda_ICU_R']-g.parameters['lambda_ICU_D'])*g.ICU[self.t]


		delta_ICU = (
			- (self.parameters["lambda_ICU_R"] + self.parameters["lambda_ICU_D"])*self.ICU[self.t]
			+ entering_icu[self.name]*(1-(summ_entering_icu-icu_cap+summ_staying_icu if summ_entering_icu-icu_cap+summ_staying_icu>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6))
		)
		self.ICU += [self.ICU[self.t]+delta_ICU*self.dt]

		# if self.ICU[-1] < 0:
		# 	print(-(self.parameters["lambda_ICU_R"] + self.parameters["lambda_ICU_D"])*self.ICU[self.t])
		# 	print(+ entering_icu[self.name])
		# 	print(+ entering_icu[self.name]*(1-(summ_entering_icu-icus if summ_entering_icu-icus>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6)))
		# 	assert(False)


	def update_D(self, m_tests, a_tests, h_cap, icu_cap):
		# For each group, calculate the entering amount
		entering_h = {}
		summ_entering_h = 0
		summ_staying_h = 0
		for n,g in self.all_groups.items():
			entering_h[n] = self.all_groups[n].flow_H(self.t)
			summ_entering_h += entering_h[n]
			summ_staying_h += (1-g.parameters['lambda_H_R']-g.parameters['lambda_H_D'])*g.H[self.t]

		entering_icu = {}
		summ_entering_icu = 0
		summ_staying_icu = 0
		for n,g in self.all_groups.items():
			entering_icu[n] = self.all_groups[n].flow_ICU(self.t)
			summ_entering_icu += entering_icu[n]
			summ_staying_icu += (1-g.parameters['lambda_ICU_R']-g.parameters['lambda_ICU_D'])*g.ICU[self.t]


		delta_D = (
			self.parameters["lambda_H_D"]*self.H[self.t]
			+ self.parameters["lambda_ICU_D"]*self.ICU[self.t]
			+ entering_icu[self.name]*((summ_entering_icu-icu_cap+summ_staying_icu if summ_entering_icu-icu_cap+summ_staying_icu>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6))
			+ entering_h[self.name]*((summ_entering_h-h_cap+summ_staying_h if summ_entering_h-h_cap+summ_staying_h>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6))
		)

		self.D += [self.D[self.t]+delta_D*self.dt]
