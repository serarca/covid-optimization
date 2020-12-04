from collections import defaultdict
import numpy as np
import pandas as pd
import math
from copy import deepcopy
import yaml

# Parameters to try
with open("../parameters/run_params.yaml") as file:
	run_params = yaml.load(file, Loader=yaml.FullLoader)
print(run_params)
groups = run_params["groups"]

if groups == "all":
	age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']
elif groups == "one":
	age_groups = ['all_age_groups']




cont = [ 'S', 'E', 'I', 'R', 'N', 'Ia', 'Ips', \
       'Ims', 'Iss', 'Rq', 'H', 'ICU', 'D' ]
activities = ['home','leisure','other','school','transport','work']


class FastDynamicalModel:
	def __init__(self, parameters, econ_params, experiment_params, dt, mixing_method, horizon, start_day, eta):
		self.parameters = deepcopy(parameters)
		self.econ_params = deepcopy(econ_params)
		self.experiment_params = deepcopy(experiment_params)
		self.dt = dt
		self.mixing_method = mixing_method
		self.original_lockdown_status = 0
		self.horizon = horizon
		self.END_DAYS = 14
		self.age_groups = list(self.parameters['seir-groups'].keys())
		self.start_day = start_day

		# Modify eta
		self.econ_params["employment_params"]["eta"] = eta
		self.econ_params["employment_params"]["nu"] = 1 - eta - self.econ_params["employment_params"]["gamma"]


		# Create groups from parameters
		self.groups = {}
		for n in parameters['seir-groups']:
			self.groups[n] = SEIR_group(parameters['seir-groups'][n], self.dt, self.mixing_method, self)



		# Fix number of beds and icus
		self.beds = self.parameters['global-parameters']['C_H']
		self.icus = self.experiment_params['icus']


		# Initialize M matrix
		self.initialize_M()

		# Initialize arrays
		self.contact_matrix = np.zeros((len(age_groups),len(age_groups)), order = "C")

		# Initialize parameters vectors
		self.p_H = self.params_vector("p_H")
		self.p_ICU = self.params_vector("p_ICU")
		self.mu = self.params_vector("mu")
		self.sigma = self.params_vector("sigma")
		self.p_Ia = self.params_vector("p_Ia")
		self.p_Ips = self.params_vector("p_Ips")
		self.p_Ims = self.params_vector("p_Ims")
		self.p_Iss = self.params_vector("p_Iss")
		self.lambda_H_R = self.params_vector("lambda_H_R")
		self.lambda_ICU_R = self.params_vector("lambda_ICU_R")
		self.lambda_H_D = self.params_vector("lambda_H_D")
		self.lambda_ICU_D = self.params_vector("lambda_ICU_D")

		# Initialize beta
		self.beta = np.zeros((len(age_groups),len(self.groups[age_groups[0]].parameters["beta"])))
		for i in range(len(age_groups)):
			for j in range(len(self.groups[age_groups[0]].parameters["beta"])):
				if j<horizon:
					self.beta[i,j] = self.groups[age_groups[i]].parameters["beta"][j+self.start_day]
				else:
					self.beta[i,j] = 0
		self.last_beta = 0
		self.last_season = ""


		# Process econ data
		econ_activities = ['transport','leisure','other']
		self.v_g = np.array([sum([self.econ_params["employment_params"]["v"][ag][act] for act in econ_activities]) for ag in age_groups])
		self.schooling_params = np.array([self.econ_params["schooling_params"][ag] for ag in age_groups])
		self.econ_cost_death = np.array([self.econ_params["econ_cost_death"][ag] for ag in age_groups])
		self.schooling_params = np.array([self.econ_params['schooling_params'][ag] for ag in age_groups])


	def params_vector(self, param):
		v = np.zeros(len(age_groups), order = "C")
		for i in range(len(age_groups)):
			v[i] = self.groups[age_groups[i]].parameters[param]
		return v


	def initialize_M(self):
		self.M_weekdays = np.zeros((len(age_groups),len(age_groups), len(activities)), order = "C")
		self.M_weekends = np.zeros((len(age_groups),len(age_groups), len(activities)), order = "C")
		self.M_new = np.zeros((len(age_groups),len(age_groups), len(activities)), order = "C")
		self.M_old = np.zeros((len(age_groups),len(age_groups), len(activities)), order = "C")
		self.M_spc = np.zeros((len(age_groups),len(age_groups), len(activities)), order = "C")
		for g1 in range(len(age_groups)):
			for g2 in range(len(age_groups)):
				for act in range(len(activities)):
					self.M_weekdays[g1,g2,act] = self.groups[age_groups[g1]].contacts_weekdays[activities[act]][age_groups[g2]]
					self.M_weekends[g1,g2,act] = self.groups[age_groups[g1]].contacts_weekends[activities[act]][age_groups[g2]]
					self.M_new[g1,g2,act] = self.groups[age_groups[g1]].contacts_new[activities[act]][age_groups[g2]]
					self.M_old[g1,g2,act] = self.groups[age_groups[g1]].contacts_old[activities[act]][age_groups[g2]]
					self.M_spc[g1,g2,act] = self.groups[age_groups[g1]].contacts_spc[activities[act]][age_groups[g2]]

	# Takes a step that is a full-open policy
	def take_end_step(self, state, season):

		return self.take_time_step(state, np.zeros(len(self.age_groups)), np.zeros(len(self.age_groups)), np.zeros((len(self.age_groups),len(activities)))+1.0, self.horizon+1, season)



	def take_time_step(self, state, m_tests, a_tests, alphas, t, season, update_contacts = True, B_H = False, B_ICU = False, B_H_perc = False, B_ICU_perc = False):

		# Store variables
		self.state = state
		self.m_tests = m_tests
		self.a_tests = a_tests
		self.alphas = alphas
		self.B_H = B_H
		self.B_ICU = B_ICU
		self.overflow_icu = 0
		self.t = t

		# Create new state
		self.new_state = np.zeros((len(age_groups),len(cont)), order = "C")

		# Update contact matric
		if update_contacts or self.beta[0,t]!=self.last_beta or season!=self.last_season:
			self.update_contact_matrix(season)
			self.last_beta = self.beta[0,t]
			self.last_season = season


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
			self.overflow_icu = overflow_icu

		elif B_ICU_perc is not False:
			summ_entering_icu = np.sum(self.flow_ICU)
			summ_staying_icu = np.sum((1-self.lambda_ICU_R-self.lambda_ICU_D)*self.state[:,cont.index("ICU")])
			overflow_icu = summ_entering_icu - self.icus + summ_staying_icu if summ_entering_icu - self.icus + summ_staying_icu>0 else 0

			self.B_ICU = overflow_icu*B_ICU_perc
			self.overflow_icu = overflow_icu



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

	def update_contact_matrix(self, season):


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
						if season == "weekdays":
							self.contact_matrix[g1,g2] += self.M_weekdays[g1,g2,act]*(self.alphas[g1,act]**self.mixing_method['param_alpha'][activities[act]])*(self.alphas[g2,act]**self.mixing_method['param_beta'][activities[act]])
						elif season == "weekends":
							self.contact_matrix[g1,g2] += self.M_weekends[g1,g2,act]*(self.alphas[g1,act]**self.mixing_method['param_alpha'][activities[act]])*(self.alphas[g2,act]**self.mixing_method['param_beta'][activities[act]])
						elif season == "new":
							self.contact_matrix[g1,g2] += self.M_new[g1,g2,act]*(self.alphas[g1,act]**self.mixing_method['param_alpha'][activities[act]])*(self.alphas[g2,act]**self.mixing_method['param_beta'][activities[act]])
						elif season == "old":
							self.contact_matrix[g1,g2] += self.M_old[g1,g2,act]*(self.alphas[g1,act]**self.mixing_method['param_alpha'][activities[act]])*(self.alphas[g2,act]**self.mixing_method['param_beta'][activities[act]])
						elif season == "spc":
							self.contact_matrix[g1,g2] += self.M_spc[g1,g2,act]*(self.alphas[g1,act]**self.mixing_method['param_alpha'][activities[act]])*(self.alphas[g2,act]**self.mixing_method['param_beta'][activities[act]])

					else:
						assert(False)

	def get_flow_H(self):
		denom = self.p_ICU + self.p_H
		denom[denom == 0] = 1e-6
		flow = self.mu*self.p_H*(self.state[:,cont.index("I")] + self.state[:,cont.index("Iss")]/denom)

		return flow


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
			- self.beta[:,self.t]*self.total_contacts
		)*self.dt

	def update_E(self):
		self.new_state[:,cont.index("E")] = self.state[:,cont.index("E")]+(
			self.beta[:,self.t]*self.total_contacts - self.sigma*self.state[:,cont.index("E")]
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
		if np.isnan(self.new_state[:,cont.index("H")]).any():
			print("joker",self.state[:,cont.index("H")])

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
		econ_activities = ["transport","leisure","other"]
		eta_activities = ["transport","leisure","other"]
		work_alpha = self.alphas[:,activities.index('work')]
		school_alpha = self.alphas[:,activities.index('school')]
		l_mean = np.mean(
			self.alphas[:,activities.index('transport')]+
			self.alphas[:,activities.index('leisure')]+
			self.alphas[:,activities.index('other')]
		)/3.0


		v_employment = (
			self.v_g*(self.econ_params["employment_params"]["nu"]*work_alpha+
				self.econ_params["employment_params"]["eta"]*l_mean+
				self.econ_params["employment_params"]["gamma"])*(
				self.new_state[:,cont.index("S")]+self.new_state[:,cont.index("I")]+self.new_state[:,cont.index("E")]+self.new_state[:,cont.index("R")]
			)*self.dt + 
			self.v_g*(self.econ_params["employment_params"]["nu"]*1.0+
				self.econ_params["employment_params"]["eta"]*1.0+
				self.econ_params["employment_params"]["gamma"])*(
				self.new_state[:,cont.index("Rq")]
			)*self.dt
		)

		v_schooling = (self.experiment_params['delta_schooling']*self.schooling_params*school_alpha*self.dt*(
				self.new_state[:,cont.index("S")]+self.new_state[:,cont.index("I")]+self.new_state[:,cont.index("E")]+self.new_state[:,cont.index("R")]
			)*self.dt +
			self.experiment_params['delta_schooling']*self.schooling_params*1.0*self.dt*(
				self.new_state[:,cont.index("Rq")]
			)*self.dt 
		)
		return np.sum(v_schooling+v_employment)

	def get_deaths(self):
		return np.sum(self.new_state[:,cont.index("D")] - self.state[:,cont.index("D")])

	def get_reward(self):
		return self.economic_value - np.sum((self.econ_cost_death+self.experiment_params["xi"])*(self.new_state[:,cont.index("D")] - self.state[:,cont.index("D")]))



class SEIR_group:
	# Time step
	def __init__(self, group_parameters, dt, mixing_method, parent):
		# Group name
		self.name = group_parameters['name']
		self.parameters = group_parameters['parameters']
		self.contacts_weekdays = group_parameters['contacts_weekdays']
		self.contacts_weekends = group_parameters['contacts_weekends']
		self.contacts_new = group_parameters['contacts_new']
		self.contacts_old = group_parameters['contacts_old']
		self.contacts_spc = group_parameters['contacts_spc']
		self.mixing_method = mixing_method
		self.parent = parent

		# Time step
		self.dt = dt


