from collections import defaultdict
from bound import Bounds
import numpy as np
import pandas as pd
import math
from gurobipy import *
import time


CONTACTS_BOUND = 100
MIPGAP = 1e-12



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
	def __init__(self, parameters, initialization, dt, time_steps, mixing_method, alphas_vec, intervals):
		self.parameters = parameters
		self.t = 0
		self.dt = dt
		self.time_steps = time_steps
		self.initialization = initialization
		self.mixing_method = mixing_method
		self.alphas_vec = alphas_vec

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

		# Initialize intervals
		for n in self.groups:
			self.groups[n].initialize_intervals(intervals)

		# Initialize IR and total_contacts
		for n in self.groups:
			self.groups[n].update_total_contacts(0, alphas_vec[0])


	def take_time_step(self, model):
		for n in self.groups:
			self.groups[n].take_time_step(model)


		# Update time
		self.t += 1



	def construct_model(self, T, m_tests, a_tests):

		t0 = time.time()


		model = Model()
		model.Params.MIPGap = MIPGAP
		z_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T):
				z_vars[group][t] = model.addVar(lb=0, ub=self.groups[group].N0*CONTACTS_BOUND ,name="z_%s_%d"%(group,t))

		m_test_vars = defaultdict(dict)
		a_test_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T):
				m_test_vars[group][t] = model.addVar(lb=0, ub=m_tests, name="Mtest_%s_%d"%(group,t))
				a_test_vars[group][t] = model.addVar(lb=0, ub=a_tests, name="Atest_%s_%d"%(group,t))

		S_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T):
				S_vars[group][t] = model.addVar(lb=0, ub=self.groups[group].N0, name="S_%s_%d"%(group,t))

		I_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T):
				I_vars[group][t] = model.addVar(lb=0, ub=self.groups[group].N0, name="I_%s_%d"%(group,t))

		IR_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T):
				IR_vars[group][t] = model.addVar(lb=0, ub=CONTACTS_BOUND, name="IR_%s_%d"%(group,t))


		# Add vars into each group
		for group in self.groups:
			for t in range(0, T):
				self.groups[group].z.append(z_vars[group][t])

		for group in self.groups:
			for t in range(0, T):
				self.groups[group].B_H.append(model.addVar(lb=0, name="B_H_%s_%d"%(group,t)))
				self.groups[group].B_ICU.append(model.addVar(lb=0, name="B_ICU_%s_%d"%(group,t)))

		for group in self.groups:
			for t in range(0, T):
				self.groups[group].m_tests.append(m_test_vars[group][t])
				self.groups[group].a_tests.append(a_test_vars[group][t])

		for group in self.groups:
			for t in range(1, T):
				self.groups[group].S.append(S_vars[group][t])

		for group in self.groups:
			for t in range(1, T):
				self.groups[group].I.append(I_vars[group][t])

		for group in self.groups:
			for t in range(1, T):
				self.groups[group].IR.append(IR_vars[group][t])



		# Add definition of S
		for name,group in self.groups.items():
			for t in range(0, T):
				model.addConstr(S_vars[name][t] == group.S[0] - group.parameters['beta']*sum([z_vars[name][tao] for tao in range(0,t)]))

		# Add definition of I
		for name,group in self.groups.items():
			for t in range(0, T):
				model.addConstr(I_vars[name][t] == (
					group.I[0]*(1-group.parameters['mu'])**t
					+ sum([(1-group.parameters['mu'])**(t-tao-1)*group.parameters['sigma']*(1-group.parameters['sigma'])**tao*group.E[0] for tao in range(0,t)])
					+ sum([(1-group.parameters['mu'])**(t-tao-1)*group.parameters['sigma']*group.parameters['beta']*
						sum([(1-group.parameters['sigma'])**(tao-k-1)*z_vars[name][k] for k in range(0,tao)]) for tao in range(0,t)])
					- sum([(1-group.parameters['mu'])**(t-tao-1)*m_test_vars[name][tao] for tao in range(0,t)])
					)
				)

		# Add definition of IR
		for n,group in self.groups.items():
			for t in range(0, T):
				model.addConstr(IR_vars[n][t] == sum([n_contacts(group, group2, self.alphas_vec[t], self.mixing_method)/group2.N0*I_vars[n2][t] for n2,group2 in self.groups.items()]))

		# Add bounds for S and IR
		for name,group in self.groups.items():
			for t in range(0,T):
				model.addConstr(S_vars[name][t] >= group.S_L[t])
				model.addConstr(S_vars[name][t] <= group.S_U[t])
				model.addConstr(IR_vars[name][t] >= group.IR_L[t])
				model.addConstr(IR_vars[name][t] <= group.IR_U[t])



		# Add McCormick envelopes
		for group in self.groups:
			for t in range(0, T):
				model.addConstr(z_vars[group][t] >= 
					self.groups[group].S_L[t]*IR_vars[group][t] 
					+ S_vars[group][t]*self.groups[group].IR_L[t]
					- self.groups[group].S_L[t]*self.groups[group].IR_L[t]
				)
				model.addConstr(z_vars[group][t] >= 
					self.groups[group].S_U[t]*IR_vars[group][t] 
					+ S_vars[group][t]*self.groups[group].IR_U[t]
					- self.groups[group].S_U[t]*self.groups[group].IR_U[t]
				)
				model.addConstr(z_vars[group][t] <= 
					self.groups[group].S_U[t]*IR_vars[group][t] 
					+ S_vars[group][t]*self.groups[group].IR_L[t]
					- self.groups[group].S_U[t]*self.groups[group].IR_L[t]
				)
				model.addConstr(z_vars[group][t] <= 
					self.groups[group].S_L[t]*IR_vars[group][t] 
					+ S_vars[group][t]*self.groups[group].IR_U[t]
					- self.groups[group].S_L[t]*self.groups[group].IR_U[t]
				)

		# Add testing constraints
		for t in range(0,T):
			model.addConstr(sum([m_test_vars[group][t] for group in self.groups])<=m_tests)
			model.addConstr(sum([a_test_vars[group][t] for group in self.groups])<=a_tests)

		# Take time steps
		for t in range(0,T):
			print(t)
			self.take_time_step(model)

		# Add bounds on B
		for name, group in self.groups.items():
			for t in range(0,T):
				model.addConstr(group.B_H[t]<=group.flow_H(t))
				model.addConstr(group.B_ICU[t]<=group.flow_ICU(t))

		for t in range(0,T):
			model.addConstr(
				sum([group.flow_H(t)-group.B_H[t] for name,group in self.groups.items()])<=
				self.beds
				- sum([(1-group.parameters["lambda_H_R"]-group.parameters["lambda_H_D"])*group.H[t] for name,group in self.groups.items()])
			)
			model.addConstr(
				sum([group.flow_ICU(t)-group.B_ICU[t] for name,group in self.groups.items()])<=
				self.icus
				- sum([(1-group.parameters["lambda_ICU_R"]-group.parameters["lambda_ICU_D"])*group.ICU[t] for name,group in self.groups.items()])
			)

		model.update()

		# Construct objective value
		objective = 0
		for t in range(1,T):
			for name, group in self.groups.items():
				objective += (
					group.economics['work_value']*(
						self.alphas_vec[t-1][name]['work']+
						group.economics['lockdown_fraction']*(1-self.alphas_vec[t-1][name]['work'])
					)*
					(group.S[t] + group.E[t] + group.R[t])
					* self.dt
				)
				# Liberate people in Rq group
				objective += group.Rq[t]*group.economics['work_value']* self.dt

		t1 = time.time()

		model.setObjective(objective,GRB.MAXIMIZE)
		model.update()
		model.optimize()

		return(model)


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
		# Leaving H and ICU
		self.B_H = []
		self.B_ICU = []


		# Contacts
		self.total_contacts = []
		self.IR = []

		# The initial population
		self.N0 = self.S[0] + self.E[0] + self.I[0]+ float(initial_conditions['R']) + float(initial_conditions['Rq'])

		# Additional vars
		self.z = []
		self.m_tests = []
		self.a_tests = []

	def update_total_contacts(self, t, alphas):
		if (len(self.total_contacts) == t):
			summ_contacts = 0
			for n,g in self.all_groups.items():
				# Set population the same as N0
				pop_g = g.N0
				new_contacts = n_contacts(self, g, alphas, self.mixing_method)
				summ_contacts += new_contacts*g.I[t]/(pop_g if pop_g!=0 else 10e-6)
			self.total_contacts.append(summ_contacts*self.S[t])
			self.IR.append(summ_contacts)
		else:
			assert(False)


	def initialize_intervals(self, intervals):
		self.S_L = intervals["S_L"][self.name]
		self.S_U = intervals["S_U"][self.name]
		self.IR_L = intervals["IR_L"][self.name]
		self.IR_U = intervals["IR_U"][self.name]


	# Attach other groups to make it easier to find variables of other groups
	def attach_other_groups(self,all_groups):
		self.all_groups = all_groups

	# Advances one time step, given the m_tests and a_tests variable
	def take_time_step(self, model):
		self.update_N(model)
		self.update_E(model)
		self.update_R(model)
		self.update_Ia(model)
		self.update_Ips(model)
		self.update_Ims(model)
		self.update_Iss(model)
		self.update_Rq(model)
		self.update_H(model)
		self.update_ICU(model)
		self.update_D(model)

		self.t += 1

	def flow_H(self, t):
		if self.parameters['p_H'] != 0.0:
			return self.parameters['mu']*self.parameters['p_H']*(self.I[t]+self.Iss[t]/(self.parameters['p_H']+self.parameters['p_ICU']))
		else:
			return 0.0

	# Gives flow of how many people flowing to ICU
	def flow_ICU(self, t):
		if self.parameters['p_ICU'] != 0.0:
			return self.parameters['mu']*self.parameters['p_ICU']*(self.I[t]+self.Iss[t]/(self.parameters['p_H']+self.parameters['p_ICU']))
		else:
			return 0.0


	def update_N(self, model):
		delta_N = (
			- self.m_tests[self.t]
			- self.a_tests[self.t]
			- self.parameters['mu']*(self.parameters['p_H'] + self.parameters['p_ICU'])*self.I[self.t]
		)

		new_var = model.addVar(lb=0, name="N_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.N[self.t]+delta_N*self.dt)
		self.N += [new_var]


	# Updates Exposed
	def update_E(self, model):
		delta_E = self.parameters['beta']*self.z[self.t] - self.parameters['sigma']*self.E[self.t]

		new_var = model.addVar(lb=0, name="E_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.E[self.t]+delta_E*self.dt)
		self.E += [new_var]


	# Updates recovered
	def update_R(self, model):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]-self.parameters["p_ICU"])*self.I[self.t] - self.a_tests[self.t]

		new_var = model.addVar(lb=0, name="R_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.R[self.t]+delta_R*self.dt)
		self.R += [new_var]

	# Updates infected in quarantine
	def update_Ia(self, model):
		delta_Ia = self.parameters['p_Ia']*self.m_tests[self.t] - self.parameters['mu']*self.Ia[self.t]

		new_var = model.addVar(lb=0, name="Ia_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.Ia[self.t]+delta_Ia*self.dt)
		self.Ia += [new_var]


	def update_Ips(self, model):
		delta_Ips = self.parameters['p_Ips']*self.m_tests[self.t] - self.parameters['mu']*self.Ips[self.t]

		new_var = model.addVar(lb=0, name="Ips_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.Ips[self.t]+delta_Ips*self.dt)
		self.Ips += [new_var]

	def update_Ims(self, model):
		delta_Ims = self.parameters['p_Ims']*self.m_tests[self.t] - self.parameters['mu']*self.Ims[self.t]

		new_var = model.addVar(lb=0, name="Ims_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.Ims[self.t]+delta_Ims*self.dt)
		self.Ims += [new_var]


	def update_Iss(self, model):
		delta_Iss = self.parameters['p_Iss']*self.m_tests[self.t] - self.parameters['mu']*self.Iss[self.t]

		new_var = model.addVar(lb=0, name="Iss_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.Iss[self.t]+delta_Iss*self.dt)
		self.Iss += [new_var]

	# Update recovered in quarentine
	def update_Rq(self, model):
		delta_Rq = (
			self.parameters['mu']*(self.Ia[self.t]+self.Ips[self.t]+self.Ims[self.t]) +
			self.parameters['lambda_H_R']*self.H[self.t] +
			self.parameters['lambda_ICU_R']*self.ICU[self.t] +
			self.a_tests[self.t]
		)

		new_var = model.addVar(lb=0, name="Rq_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.Rq[self.t]+delta_Rq*self.dt)
		self.Rq += [new_var]

	def update_H(self, model):
		delta_H = (
			- (self.parameters["lambda_H_R"] + self.parameters["lambda_H_D"])*self.H[self.t]
			+ self.flow_H(self.t)
			- self.B_H[self.t]
		)

		new_var = model.addVar(lb=0, name="H_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.H[self.t]+delta_H*self.dt)
		self.H += [new_var]




	def update_ICU(self, model):
		delta_ICU = (
			- (self.parameters["lambda_ICU_R"] + self.parameters["lambda_ICU_D"])*self.ICU[self.t]
			+ self.flow_ICU(self.t)
			- self.B_ICU[self.t]
		)

		new_var = model.addVar(lb=0, name="ICU_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.ICU[self.t]+delta_ICU*self.dt)
		self.ICU += [new_var]


	def update_D(self, model):
		delta_D = (
			self.parameters["lambda_H_D"]*self.H[self.t]
			+ self.parameters["lambda_ICU_D"]*self.ICU[self.t]
			+ self.B_H[self.t]
			+ self.B_ICU[self.t]
		)

		new_var = model.addVar(lb=0, name="D_%s_%d"%(self.name,self.t+1))
		model.addConstr(new_var == self.D[self.t]+delta_D*self.dt)
		self.D += [new_var]






