from collections import defaultdict
import numpy as np
import pandas as pd
import math

age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29','age_group_30_39', 'age_group_40_49', 'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]
activities = ['home','leisure','other','school','transport','work']


class FastDynamicalModel:
	def __init__(self, parameters, dt, mixing_method):
		self.parameters = parameters
		self.dt = dt
		self.mixing_method = mixing_method

		# Create groups from parameters
		self.groups = {}
		for n in parameters['seir-groups']:
			self.groups[n] = SEIR_group(parameters['seir-groups'][n], self.dt, self.mixing_method, self)


		# Fix number of beds and icus
		self.beds = self.parameters['global-parameters']['C_H']
		self.icus = self.parameters['global-parameters']['C_ICU']

		# Initialize M matrix
		self.initialize_M()

		# Initialize arrays
		self.contact_matrix = np.zeros((len(age_groups),len(age_groups)), order = "C")

		# Initialize parameters vectors
		self.p_H = self.params_vector("p_H")
		self.p_ICU = self.params_vector("p_ICU")
		self.mu = self.params_vector("mu")
		self.sigma = self.params_vector("sigma")
		self.beta = self.params_vector("beta")
		self.p_Ia = self.params_vector("p_Ia")
		self.p_Ips = self.params_vector("p_Ips")
		self.p_Ims = self.params_vector("p_Ims")
		self.p_Iss = self.params_vector("p_Iss")
		self.lambda_H_R = self.params_vector("lambda_H_R")
		self.lambda_ICU_R = self.params_vector("lambda_ICU_R")
		self.lambda_H_D = self.params_vector("lambda_H_D")
		self.lambda_ICU_D = self.params_vector("lambda_ICU_D")
		self.work_value = self.params_vector("work_value")
		self.lockdown_fraction = self.params_vector("lockdown_fraction")
		self.death_value = self.params_vector("death_value")

	def params_vector(self, param):
		v = np.zeros(len(age_groups), order = "C")
		for i in range(len(age_groups)):
			v[i] = self.groups[age_groups[i]].parameters[param]
		return v


	def initialize_M(self):
		self.M = np.zeros((len(age_groups),len(age_groups), len(activities)), order = "C")
		for g1 in range(len(age_groups)):
			for g2 in range(len(age_groups)):
				for act in range(len(activities)):
					self.M[g1,g2,act] = self.groups[age_groups[g1]].contacts[activities[act]][age_groups[g2]]


	def take_time_step(self, state, m_tests, a_tests, alphas, update_contacts = True, B_H = False, B_ICU = False, B_H_perc = False, B_ICU_perc = False):

		# Store variables
		self.state = state
		self.m_tests = m_tests
		self.a_tests = a_tests
		self.alphas = alphas
		self.B_H = B_H
		self.B_ICU = B_ICU

		# Create new state
		self.new_state = np.zeros((len(age_groups),len(cont)), order = "C")

		# Update contact matric
		if update_contacts:
			self.update_contact_matrix()

		# Update total contacts
		pop = self.state[:,cont.index("N")] + self.state[:,cont.index("Rq")]
		pop[pop==0.0] = 1e-6
		self.total_contacts = self.state[:,cont.index("S")]*np.matmul(
			self.contact_matrix,
			self.state[:,cont.index("I")]/pop
		)

		# Construct flow variables
		self.flow_H = self.get_flow_H()
		self.flow_ICU = self.get_flow_ICU()

		# # Construct prorrated bouncing
		if ((B_H is False) and (B_H_perc is False)):
			# Bouncing for H
			summ_entering_h = np.sum(self.flow_H)
			summ_staying_h = np.sum((1-self.lambda_H_R-self.lambda_H_D)*self.state[:,cont.index("H")])
			overflow_h = summ_entering_h - self.beds + summ_staying_h if summ_entering_h - self.beds + summ_staying_h>0 else 0
			summ_entering_h = summ_entering_h if summ_entering_h!=0 else 1e-6
			
			self.B_H = self.flow_H*overflow_h/summ_entering_h

		elif B_H_perc is not False:
			# Bouncing for H
			summ_entering_h = np.sum(self.flow_H)
			summ_staying_h = np.sum((1-self.lambda_H_R-self.lambda_H_D)*self.state[:,cont.index("H")])
			overflow_h = summ_entering_h - self.beds + summ_staying_h if summ_entering_h - self.beds + summ_staying_h>0 else 0

			self.B_H = overflow_h*B_H_perc

		# # Construct prorrated bouncing
		if ((B_ICU is False) and (B_ICU_perc is False)):
			# Bouncing for ICU
			summ_entering_icu = np.sum(self.flow_ICU)
			summ_staying_icu = np.sum((1-self.lambda_ICU_R-self.lambda_ICU_D)*self.state[:,cont.index("ICU")])
			overflow_icu = summ_entering_icu - self.icus + summ_staying_icu if summ_entering_icu - self.icus + summ_staying_icu>0 else 0
			summ_entering_icu = summ_entering_icu if summ_entering_icu!=0 else 1e-6
			
			self.B_ICU = self.flow_ICU*overflow_icu/summ_entering_icu

		elif B_ICU_perc is not False:
			summ_entering_icu = np.sum(self.flow_ICU)
			summ_staying_icu = np.sum((1-self.lambda_ICU_R-self.lambda_ICU_D)*self.state[:,cont.index("ICU")])
			overflow_icu = summ_entering_icu - self.icus + summ_staying_icu if summ_entering_icu - self.icus + summ_staying_icu>0 else 0

			self.B_ICU = overflow_icu*B_ICU_perc



		# Create a population variable, this is useful to avoid division by zero
		self.pop = np.copy(self.state[:,cont.index("N")])
		self.pop[self.pop == 0.0]


		# Update states
		self.update_N()
		self.update_S()
		self.update_E()
		self.update_I()
		self.update_R()
		self.update_Ia()
		self.update_Ips()
		self.update_Ims()
		self.update_Iss()
		self.update_Rq()
		self.update_H()
		self.update_ICU()
		self.update_D()

		# # Get new economic values
		self.economic_value = self.get_economic_value()
		self.deaths = self.get_deaths()
		self.reward = self.get_reward()


		return self.new_state, {
			"economic_value":self.economic_value,
			"deaths":self.deaths,
			"reward":self.reward,
		}

		# return {
		# 	"new_state": self.new_state,
		# 	"economic_value": self.economic_value,
		# 	"deaths": self.deaths,
		# 	"reward": self.reward,
		# }

	def update_contact_matrix(self):
		for g1 in range(len(age_groups)):
			for g2 in range(len(age_groups)):
				self.contact_matrix[g1,g2] = 0
				for act in range(len(activities)):
					if self.mixing_method['name'] == "maxmin":
						self.contact_matrix[g1,g2] += self.M[g1,g2,act]*(
								(self.alphas[g1,act]*math.exp(self.alphas[g1,act]*self.mixing_method['param']) + self.alphas[g2,act]*math.exp(self.alphas[g2,act]*self.mixing_method['param']))
								/(math.exp(self.alphas[g1,act]*self.mixing_method['param'])+math.exp(self.alphas[g2,act]*self.mixing_method['param']))
							)
					elif self.mixing_method['name'] == "mult":
						self.contact_matrix[g1,g2] += self.M[g1,g2,act]*(self.alphas[g1,act]**self.mixing_method['param_alpha'])*(self.alphas[g2,act]**self.mixing_method['param_beta'])
					else:
						assert(False)

	def get_flow_H(self):
		denom = self.p_ICU + self.p_H
		denom[denom == 0] = 1e-6
		return self.mu*self.p_H*(self.state[:,cont.index("I")] + self.state[:,cont.index("Iss")]/denom)

	def get_flow_ICU(self):
		denom = self.p_ICU + self.p_H
		denom[denom == 0] = 1e-6
		return self.mu*self.p_ICU*(self.state[:,cont.index("I")] + self.state[:,cont.index("Iss")]/denom)

	def update_N(self):
		self.new_state[:,cont.index("N")] = self.state[:,cont.index("N")]+(
			- self.m_tests*self.state[:,cont.index("I")]/self.pop
			- self.a_tests*self.state[:,cont.index("R")]/self.pop
			- self.mu*(self.p_H + self.p_ICU)*self.state[:,cont.index("I")]
		)*self.dt

	def update_S(self):
		self.new_state[:,cont.index("S")] = self.state[:,cont.index("S")]+(
			- self.beta*self.total_contacts
		)*self.dt

	def update_E(self):
		self.new_state[:,cont.index("E")] = self.state[:,cont.index("E")]+(
			self.beta*self.total_contacts - self.sigma*self.state[:,cont.index("E")]
		)*self.dt

	def update_I(self):
		self.new_state[:,cont.index("I")] = self.state[:,cont.index("I")]+(
			self.sigma*self.state[:,cont.index("E")] - self.mu*self.state[:,cont.index("I")] - self.m_tests*self.state[:,cont.index("I")]/self.pop
		)*self.dt

	def update_R(self):
		self.new_state[:,cont.index("R")] = self.state[:,cont.index("R")]+(
			self.mu*(1-self.p_H-self.p_ICU)*self.state[:,cont.index("I")] - self.a_tests*self.state[:,cont.index("R")]/self.pop
		)*self.dt

	def update_Ia(self):
		self.new_state[:,cont.index("Ia")] = self.state[:,cont.index("Ia")]+(
			self.p_Ia*self.m_tests*self.state[:,cont.index("I")]/self.pop - self.mu*self.state[:,cont.index("Ia")]
		)*self.dt

	def update_Ips(self):
		self.new_state[:,cont.index("Ips")] = self.state[:,cont.index("Ips")]+(
			self.p_Ips*self.m_tests*self.state[:,cont.index("I")]/self.pop - self.mu*self.state[:,cont.index("Ips")]
		)*self.dt

	def update_Ims(self):
		self.new_state[:,cont.index("Ims")] = self.state[:,cont.index("Ims")]+(
			self.p_Ims*self.m_tests*self.state[:,cont.index("I")]/self.pop - self.mu*self.state[:,cont.index("Ims")]
		)*self.dt

	def update_Iss(self):
		self.new_state[:,cont.index("Iss")] = self.state[:,cont.index("Iss")]+(
			self.p_Iss*self.m_tests*self.state[:,cont.index("I")]/self.pop - self.mu*self.state[:,cont.index("Iss")]
		)*self.dt

	def update_Rq(self):
		self.new_state[:,cont.index("Rq")] = self.state[:,cont.index("Rq")]+(
			self.mu*(self.state[:,cont.index("Ia")] + self.state[:,cont.index("Ips")] + self.state[:,cont.index("Ims")]) +
			self.lambda_H_R*self.state[:,cont.index("H")] + 
			self.lambda_ICU_R*self.state[:,cont.index("ICU")] +
			self.a_tests*self.state[:,cont.index("R")]/self.pop
		)*self.dt

	def update_H(self):
		tol = 1e-7
		self.new_state[:,cont.index("H")] = self.state[:,cont.index("H")]+(
			-(self.lambda_H_R + self.lambda_H_D)*self.state[:,cont.index("H")]
			+(1-tol)*self.flow_H
			-self.B_H
		)*self.dt

	def update_ICU(self):
		tol = 1e-7
		self.new_state[:,cont.index("ICU")] = self.state[:,cont.index("ICU")]+(
			-(self.lambda_ICU_R + self.lambda_ICU_D)*self.state[:,cont.index("ICU")]
			+(1-tol)*self.flow_ICU
			-self.B_ICU
		)*self.dt


	def update_D(self):
		self.new_state[:,cont.index("D")] = self.state[:,cont.index("D")]+(
			self.lambda_H_D*self.state[:,cont.index("H")]
			+ self.lambda_ICU_D*self.state[:,cont.index("ICU")]
			+ self.B_H
			+ self.B_ICU
		)*self.dt

	def get_economic_value(self):
		values = self.work_value*(self.alphas[:,activities.index('work')]+
			self.lockdown_fraction*(1-self.alphas[:,activities.index('work')])
		)*(
		self.new_state[:,cont.index("S")]+self.new_state[:,cont.index("E")]+self.new_state[:,cont.index("R")]
		)*self.dt

		values += self.new_state[:,cont.index("Rq")]*self.work_value*self.dt
		return np.sum(values)

	def get_deaths(self):
		return np.sum(self.new_state[:,cont.index("D")] - self.state[:,cont.index("D")])

	def get_reward(self):
		return self.economic_value - np.sum(self.death_value*(self.new_state[:,cont.index("D")] - self.state[:,cont.index("D")]))



class SEIR_group:
	# Time step
	def __init__(self, group_parameters, dt, mixing_method, parent):
		# Group name
		self.name = group_parameters['name']
		self.parameters = group_parameters['parameters']
		self.contacts = group_parameters['contacts']
		self.parameters['work_value'] = group_parameters['economics']['work_value']
		self.parameters['lockdown_fraction'] = group_parameters['economics']['lockdown_fraction']
		self.parameters['death_value'] = group_parameters['economics']['death_value']
		self.mixing_method = mixing_method
		self.parent = parent

		# Time step
		self.dt = dt


