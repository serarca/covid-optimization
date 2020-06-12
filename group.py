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


class DynamicalModel:
	def __init__(self, parameters, initialization, dt, time_steps, mixing_method, extra_data = False):
		self.parameters = parameters
		self.t = 0
		self.dt = dt
		self.time_steps = time_steps
		self.initialization = initialization
		self.mixing_method = mixing_method
		self.extra_data = extra_data

		# Create groups from parameters
		self.groups = {}
		for n in parameters['seir-groups']:
			self.groups[n] = SEIR_group(parameters['seir-groups'][n], initialization[n], self.dt, self.mixing_method, self.time_steps, self)

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

		# Initialize number of contacts
		if extra_data:
			self.n_contacts = [{g_name1:{g_name2:float('inf') for g_name2 in self.groups} for g_name1 in self.groups} for i in range(self.time_steps)]

	def take_time_step(self, m_tests, a_tests, alphas, B_H = False, B_ICU = False):
		for n in self.groups:
			self.groups[n].update_total_contacts(self.t, alphas)
		if B_H and B_ICU:
			# Verify that the bouncing variables satisfy the required bounds
			for n,g in self.groups.items():
				assert(B_H[n]<=g.flow_H(self.t))
				assert(B_ICU[n]<=g.flow_ICU(self.t))

			assert(
				sum([group.flow_H(self.t)-B_H[name] for name,group in self.groups.items()])<=
				self.beds
				- sum([(1-group.parameters["lambda_H_R"]-group.parameters["lambda_H_D"])*group.H[self.t] for name,group in self.groups.items()])
			)
			assert(
				sum([group.flow_ICU(self.t)-B_ICU[name] for name,group in self.groups.items()])<=
				self.icus
				- sum([(1-group.parameters["lambda_ICU_R"]-group.parameters["lambda_ICU_D"])*group.ICU[self.t] for name,group in self.groups.items()])
			)

			for n in self.groups:
				self.groups[n].take_time_step(m_tests[n], a_tests[n], self.beds, self.icus, B_H[n], B_ICU[n])
		else:
			for n in self.groups:
				self.groups[n].take_time_step(m_tests[n], a_tests[n], self.beds, self.icus, False, False)

				


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

	# Reset the overall simulation time
	def reset_time(self, new_time):
		"""Resets the current time of the simulation to an earlier time point"""

		if(new_time > self.t):
			assert(False)

		# reset the time in each group
		for n in self.groups:
			self.groups[n].reset_time(new_time)

		# reset internal calculations of econ values, deaths, rewards
		self.economic_values = self.economic_values[0:new_time+1]
		self.deaths = self.deaths[0:new_time+1]
		self.rewards = self.rewards[0:new_time+1]

		self.t = new_time

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

	def write_state(self, t, X):
		for group_name in X.keys():
			self.groups[group_name].S[t] = X[group_name]['S']
			self.groups[group_name].E[t] = X[group_name]['E']
			self.groups[group_name].I[t] = X[group_name]['I']
			self.groups[group_name].R[t] = X[group_name]['R']
			self.groups[group_name].N[t] = X[group_name]['N']
			self.groups[group_name].Ia[t] = X[group_name]['Ia']
			self.groups[group_name].Ips[t] = X[group_name]['Ips']
			self.groups[group_name].Ims[t] = X[group_name]['Ims']
			self.groups[group_name].Iss[t] = X[group_name]['Iss']
			self.groups[group_name].Rq[t] = X[group_name]['Rq']
			self.groups[group_name].H[t] = X[group_name]['H']
			self.groups[group_name].ICU[t] = X[group_name]['ICU']
			self.groups[group_name].D[t] = X[group_name]['D']
		return 1


	# Returns state but in OpenAIGym Format
	def get_normalized_state(self, t):
		norm_state = np.array([[
			group.S[t]/self.total_population,
			group.E[t]/self.total_population,
			group.I[t]/self.total_population,
			group.R[t]/self.total_population,
			(group.Ia[t] + group.Ips[t] + group.Ims[t])/self.total_population,
			group.Iss[t]/self.total_population,
			group.Rq[t]/self.total_population,
			group.H[t]/self.total_population,
			group.ICU[t]/self.total_population,
			group.D[t]/self.total_population
			] for name,group in self.groups.items()])
		return norm_state.flatten()


	# Returns state but in OpenAIGym Format
	def get_normalized_partial_state(self, t):
		norm_state = np.array([[
			(group.Ia[t] + group.Ips[t] + group.Ims[t])/self.total_population,
			group.Iss[t]/self.total_population,
			group.Rq[t]/self.total_population,
			group.H[t]/self.total_population,
			group.ICU[t]/self.total_population,
			group.D[t]/self.total_population
			] for name,group in self.groups.items()])
		return norm_state.flatten()

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

	def get_pandas_summary(self):
		d = {
			"S": [sum([self.groups[g].S[t] for g in self.groups ]) for t in range(self.t+1)],
			"E": [sum([self.groups[g].E[t] for g in self.groups ]) for t in range(self.t+1)],
			"I": [sum([self.groups[g].I[t] for g in self.groups ]) for t in range(self.t+1)],
			"R": [sum([self.groups[g].R[t] for g in self.groups ]) for t in range(self.t+1)],
			"N": [sum([self.groups[g].N[t] for g in self.groups ]) for t in range(self.t+1)],
			"Ia": [sum([self.groups[g].Ia[t] for g in self.groups ]) for t in range(self.t+1)],
			"Ips": [sum([self.groups[g].Ips[t] for g in self.groups ]) for t in range(self.t+1)],
			"Ims": [sum([self.groups[g].Ims[t] for g in self.groups ]) for t in range(self.t+1)],
			"Iss": [sum([self.groups[g].Iss[t] for g in self.groups ]) for t in range(self.t+1)],
			"Rq": [sum([self.groups[g].Rq[t] for g in self.groups ]) for t in range(self.t+1)],
			"H": [sum([self.groups[g].H[t] for g in self.groups ]) for t in range(self.t+1)],
			"ICU": [sum([self.groups[g].ICU[t] for g in self.groups ]) for t in range(self.t+1)],
			"D": [sum([self.groups[g].D[t] for g in self.groups ]) for t in range(self.t+1)],
		}
		return pd.DataFrame(d)

class SEIR_group:
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
		self.IR = []


	def update_total_contacts(self, t, alphas):
		if (len(self.total_contacts) == t):
			summ_contacts = 0
			for n,g in self.all_groups.items():
				pop_g = g.N[t] + g.Rq[t]
				new_contacts = n_contacts(self, g, alphas, self.mixing_method)
				summ_contacts += new_contacts*g.I[t]/(pop_g if pop_g!=0 else 10e-6)
				if self.parent.extra_data:
					self.parent.n_contacts[t][self.name][g.name] = new_contacts
			self.total_contacts.append(summ_contacts*self.S[t])
			self.IR.append(summ_contacts)

		else:
			print("t = ", t, "len of self.total_contacts = ", len(self.total_contacts))
			assert(False)


	# Attach other groups to make it easier to find variables of other groups
	def attach_other_groups(self,all_groups):
		self.all_groups = all_groups

	# Advances one time step, given the m_tests and a_tests variable
	def take_time_step(self, m_tests, a_tests, h_cap, icu_cap, B_H, B_ICU):
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
		self.update_H(m_tests, a_tests, h_cap, icu_cap, B_H)
		self.update_ICU(m_tests, a_tests, h_cap, icu_cap, B_ICU)
		self.update_D(m_tests, a_tests, h_cap, icu_cap, B_H, B_ICU)

		self.t += 1

	# Reset the time to a past time
	def reset_time(self, new_time):
		if(new_time > self.t):
			assert(False)
		self.S = self.S[0:new_time+1]
		self.E = self.E[0:new_time+1]
		self.I = self.I[0:new_time+1]
		self.R = self.R[0:new_time+1]
		self.N = self.N[0:new_time+1]
		self.Ia = self.Ia[0:new_time+1]
		self.Ips = self.Ips[0:new_time+1]
		self.Ims = self.Ims[0:new_time+1]
		self.Iss = self.Iss[0:new_time+1]
		self.Rq = self.Rq[0:new_time+1]
		self.H = self.H[0:new_time+1]
		self.ICU = self.ICU[0:new_time+1]
		self.D = self.D[0:new_time+1]
		self.total_contacts = self.total_contacts[0:new_time]
		
		self.t = new_time


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
			- m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
			- a_tests*self.R[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
			- self.parameters['mu']*(self.parameters['p_H'] + self.parameters['p_ICU'])*self.I[self.t]
		)
		self.N += [self.N[self.t]+delta_N*self.dt]

	# Updates S
	def update_S(self, m_tests, a_tests):
		delta_S = -self.parameters['beta']*self.total_contacts[self.t]
		self.S += [self.S[self.t]+delta_S*self.dt]

	# Updates Exposed
	def update_E(self, m_tests, a_tests):
		delta_E = self.parameters['beta']*self.total_contacts[self.t] - self.parameters['sigma']*self.E[self.t]
		self.E += [self.E[self.t]+delta_E*self.dt]


	# Updates infected
	def update_I(self, m_tests, a_tests):
		delta_I = self.parameters['sigma']*self.E[self.t] - self.parameters['mu']*self.I[self.t] - m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
		self.I += [self.I[self.t]+delta_I*self.dt]


	# Updates recovered
	def update_R(self, m_tests, a_tests):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]-self.parameters["p_ICU"])*self.I[self.t] - a_tests*self.R[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
		self.R += [self.R[self.t]+delta_R*self.dt]


	# Updates infected in quarantine
	def update_Ia(self, m_tests, a_tests):
		delta_Ia = self.parameters['p_Ia']*m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6) - self.parameters['mu']*self.Ia[self.t]
		self.Ia += [self.Ia[self.t]+delta_Ia*self.dt]

	def update_Ips(self, m_tests, a_tests):
		delta_Ips = self.parameters['p_Ips']*m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6) - self.parameters['mu']*self.Ips[self.t]
		self.Ips += [self.Ips[self.t]+delta_Ips*self.dt]

	def update_Ims(self, m_tests, a_tests):
		delta_Ims = self.parameters['p_Ims']*m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6) - self.parameters['mu']*self.Ims[self.t]
		self.Ims += [self.Ims[self.t]+delta_Ims*self.dt]

	def update_Iss(self, m_tests, a_tests):
		delta_Iss = self.parameters['p_Iss']*m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6) - self.parameters['mu']*self.Iss[self.t]
		self.Iss += [self.Iss[self.t]+delta_Iss*self.dt]


	# Update recovered in quarentine
	def update_Rq(self, m_tests, a_tests):
		delta_Rq = (
			self.parameters['mu']*(self.Ia[self.t]+self.Ips[self.t]+self.Ims[self.t]) +
			self.parameters['lambda_H_R']*self.H[self.t] +
			self.parameters['lambda_ICU_R']*self.ICU[self.t] +
			a_tests*self.R[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
		)
		self.Rq += [self.Rq[self.t]+delta_Rq*self.dt]


	def update_H(self, m_tests, a_tests, h_cap, icu_cap, B_H):
		
		# For each group, calculate the entering amount
		entering_h = {}
		summ_entering_h = 0
		summ_staying_h = 0
		for n,g in self.all_groups.items():
			entering_h[n] = self.all_groups[n].flow_H(self.t)
			summ_entering_h += entering_h[n]
			summ_staying_h += (1-g.parameters['lambda_H_R']-g.parameters['lambda_H_D'])*g.H[self.t]
		
		if B_H is False:
			B_H = entering_h[self.name]*(summ_entering_h-h_cap+summ_staying_h if summ_entering_h-h_cap+summ_staying_h>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6)

		delta_H = (
			- (self.parameters["lambda_H_R"] + self.parameters["lambda_H_D"])*self.H[self.t]
			+ entering_h[self.name]
			- B_H
		)
		self.H += [self.H[self.t]+delta_H*self.dt]


	def update_ICU(self, m_tests, a_tests, h_cap, icu_cap, B_ICU):
		# For each group, calculate the entering amount
		entering_icu = {}
		summ_entering_icu = 0
		summ_staying_icu = 0
		for n,g in self.all_groups.items():
			entering_icu[n] = self.all_groups[n].flow_ICU(self.t)
			summ_entering_icu += entering_icu[n]
			summ_staying_icu += (1-g.parameters['lambda_ICU_R']-g.parameters['lambda_ICU_D'])*g.ICU[self.t]

		if B_ICU is False:
			B_ICU = entering_icu[self.name]*(summ_entering_icu-icu_cap+summ_staying_icu if summ_entering_icu-icu_cap+summ_staying_icu>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6)

		delta_ICU = (
			- (self.parameters["lambda_ICU_R"] + self.parameters["lambda_ICU_D"])*self.ICU[self.t]
			+ entering_icu[self.name]
			- B_ICU
		)
		self.ICU += [self.ICU[self.t]+delta_ICU*self.dt]



	def update_D(self, m_tests, a_tests, h_cap, icu_cap, B_H, B_ICU):
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

		if B_H is False:
			B_H = entering_h[self.name]*(summ_entering_h-h_cap+summ_staying_h if summ_entering_h-h_cap+summ_staying_h>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6)
		if B_ICU is False:
			B_ICU = entering_icu[self.name]*(summ_entering_icu-icu_cap+summ_staying_icu if summ_entering_icu-icu_cap+summ_staying_icu>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6)


		delta_D = (
			self.parameters["lambda_H_D"]*self.H[self.t]
			+ self.parameters["lambda_ICU_D"]*self.ICU[self.t]
			+ B_H
			+ B_ICU
		)

		self.D += [self.D[self.t]+delta_D*self.dt]
