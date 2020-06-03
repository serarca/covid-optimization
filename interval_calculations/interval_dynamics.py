from collections import defaultdict
from bound import Bounds
import numpy as np
import pandas as pd
import math
from gurobipy import *
import time


CONTACTS_BOUND = 100
MIPGAP = 1e-12

def solution_values(z_vars, m_test_vars, S_vars, I_vars, IR_vars):
	solution_v = {
					"z":defaultdict(dict),
					"m":defaultdict(dict),
					"S":defaultdict(dict),
					"I":defaultdict(dict),
					"IR":defaultdict(dict),
			}
	for group in z_vars:
		for t in z_vars[group]:
			solution_v["z"][group][t] = z_vars[group][t].x
			solution_v["m"][group][t] = m_test_vars[group][t].x
			solution_v["S"][group][t] = S_vars[group][t].x
			solution_v["I"][group][t] = I_vars[group][t].x
			solution_v["IR"][group][t] = IR_vars[group][t].x
	return solution_v

def set_start(z_vars, m_test_vars, S_vars, I_vars, IR_vars, start):
	for group in start["z"]:
		for t in start["z"][group]:
			z_vars[group][t].start = start["z"][group][t] 
			m_test_vars[group][t].start = start["m"][group][t] 
			S_vars[group][t].start = start["S"][group][t] 
			I_vars[group][t].start = start["I"][group][t] 
			IR_vars[group][t].start = start["IR"][group][t] 

def set_bounds(model, z_vars, m_test_vars, S_vars, I_vars, IR_vars, force):
	for group in force["z"]:
		for t in force["z"][group]:
			model.addConstr(z_vars[group][t] == force["z"][group][t])
			model.addConstr(m_test_vars[group][t] == force["m"][group][t])
		for t in force["S"][group]:
			model.addConstr(S_vars[group][t] == force["S"][group][t])
			model.addConstr(I_vars[group][t] == force["I"][group][t])
			model.addConstr(IR_vars[group][t] == force["IR"][group][t])




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


class DynamicalModelInterval:
	def __init__(self, parameters, initialization, dt, time_steps, mixing_method, alphas_vec):
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

		# Initialize bounds
		for n in self.groups:
			self.groups[n].initialize_bounds(alphas_vec[0])


	def take_time_step(self, m_tests, a_tests, alphas):
		for n in self.groups:
			self.groups[n].update_total_contacts(self.t, alphas)
		for n in self.groups:
			self.groups[n].take_time_step(m_tests[n], a_tests[n])


		# Update time
		self.t += 1

	# Simulates the dynamics given a vector of molecular tests, atomic tests and alphas
	def simulate(self, m_tests_vec, a_tests_vec, alphas_vec):
		for t in range(self.time_steps):
			self.take_time_step(m_tests_vec[t], a_tests_vec[t], alphas_vec[t])


	def construct_model(self, T, m_tests, a_tests, warm_start = False, force = False):

		t0 = time.time()

		# This will store the solution of the model to be able to use it as a warm start
		solution = {
			"S_L":{
				"z":defaultdict(dict),
				"m":defaultdict(dict),
				"S":defaultdict(dict),
				"I":defaultdict(dict),
				"IR":defaultdict(dict),
			},
			"S_U":{
				"z":defaultdict(dict),
				"m":defaultdict(dict),
				"S":defaultdict(dict),
				"I":defaultdict(dict),
				"IR":defaultdict(dict),
			},
			"IR_L":{
				"z":defaultdict(dict),
				"m":defaultdict(dict),
				"S":defaultdict(dict),
				"I":defaultdict(dict),
				"IR":defaultdict(dict),
			},
			"IR_U":{
				"z":defaultdict(dict),
				"m":defaultdict(dict),
				"S":defaultdict(dict),
				"I":defaultdict(dict),
				"IR":defaultdict(dict),
			}
		}

		model = Model()
		model.Params.MIPGap = MIPGAP
		z_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T):
				z_vars[group][t] = model.addVar(lb=0, ub=self.groups[group].N0*CONTACTS_BOUND ,name="z_%s_%d"%(group,t))

		m_test_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T):
				m_test_vars[group][t] = model.addVar(lb=0, ub=m_tests ,name="Mtest_%s_%d"%(group,t))

		S_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T+1):
				S_vars[group][t] = model.addVar(lb=0, ub=self.groups[group].N0, name="S_%s_%d"%(group,t))

		I_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T+1):
				I_vars[group][t] = model.addVar(lb=0, ub=self.groups[group].N0, name="I_%s_%d"%(group,t))

		IR_vars = defaultdict(dict)
		for group in self.groups:
			for t in range(0, T+1):
				IR_vars[group][t] = model.addVar(lb=0, ub=CONTACTS_BOUND, name="IR_%s_%d"%(group,t))


		# Add definition of S
		for name,group in self.groups.items():
			for t in range(0, T+1):
				model.addConstr(S_vars[name][t] == group.S[0] - group.parameters['beta']*sum([z_vars[name][tao] for tao in range(0,t)]))

		# Add definition of I
		for name,group in self.groups.items():
			for t in range(0, T+1):
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
			for t in range(0, T+1):
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

		# Force variables
		if force:
			set_bounds(model,z_vars, m_test_vars, S_vars, I_vars, IR_vars, force)
			print("Forced")
			print(T)



		model.update()


		t1 = time.time()
		# Get lower bound S
		for name,group in self.groups.items():
			model.setObjective(S_vars[name][T],GRB.MINIMIZE)
			if warm_start:
				set_start(z_vars, m_test_vars, S_vars, I_vars, IR_vars, warm_start["S_L"])
			model.update()
			model.setParam( 'OutputFlag', False )

			model.Params.NumericFocus = 3
			model.optimize()
			if T==3:
				model.write("out.lp")
			if model.status == GRB.Status.INFEASIBLE:
				print(name)
				print("S_L")
				print(T)
				model.computeIIS()
				model.write("out.ilp")
				assert(False)

			lb_S = max(0,model.objVal)
			if T == 0:
				assert(np.abs((group.S_L[0] - lb_S)/lb_S)<1e-6)
			else:
				group.S_L.append(lb_S)
			solution["S_L"] = solution_values(z_vars, m_test_vars, S_vars, I_vars, IR_vars)

		# Get upper bound S
		for name,group in self.groups.items():
			model.setObjective(S_vars[name][T],GRB.MAXIMIZE)
			if warm_start:
				set_start(z_vars, m_test_vars, S_vars, I_vars, IR_vars, warm_start["S_U"])
			model.update()
			model.setParam( 'OutputFlag', False )
			if T == 3 and name == "age_group_50_59":
				model.setParam( 'OutputFlag', True )
			model.Params.NumericFocus = 3
			model.optimize()
			if model.status == GRB.Status.INFEASIBLE:
				print(name)
				print("S_U")
				assert(False)
			ub_S = model.objVal
			if T == 0:
				assert(np.abs((group.S_U[0] - ub_S)/ub_S)<1e-6)
			else:
				group.S_U.append(ub_S)
			solution["S_U"] = solution_values(z_vars, m_test_vars, S_vars, I_vars, IR_vars)


		# Get lower bound IR
		for name,group in self.groups.items():
			model.setObjective(IR_vars[name][T],GRB.MINIMIZE)
			if warm_start:
				set_start(z_vars, m_test_vars, S_vars, I_vars, IR_vars, warm_start["IR_L"])
			model.update()
			model.setParam( 'OutputFlag', False )
			model.Params.NumericFocus = 3
			model.optimize()
			if model.status == GRB.Status.INFEASIBLE:
				print(name)
				print("IR_L")
				assert(False)
			lb_IR = max(0,model.objVal)
			if T == 0:
				assert(np.abs((group.IR_L[0] - lb_IR)/lb_IR)<1e-6)
			else:
				group.IR_L.append(lb_IR)
			solution["IR_L"] = solution_values(z_vars, m_test_vars, S_vars, I_vars, IR_vars)

		# Get upper bound IR
		for name,group in self.groups.items():
			model.setObjective(IR_vars[name][T],GRB.MAXIMIZE)
			if warm_start:
				set_start(z_vars, m_test_vars, S_vars, I_vars, IR_vars, warm_start["S_U"])
			model.update()
			model.setParam( 'OutputFlag', False )
			model.Params.NumericFocus = 3
			model.optimize()
			if model.status == GRB.Status.INFEASIBLE:
				print(name)
				print("IR_U")
				assert(False)
			ub_IR = model.objVal
			if T == 0:
				assert(np.abs((group.IR_U[0] - ub_IR)/ub_IR)<1e-6)
			else:
				group.IR_U.append(ub_IR)
			solution["IR_U"] = solution_values(z_vars, m_test_vars, S_vars, I_vars, IR_vars)
		t2 = time.time()

		print(T,t1-t0,t2-t1)
		return(solution)



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
		self.old_S = [float(initial_conditions['S'])]
		# Exposed (unquarantined)
		self.E = [float(initial_conditions['E'])]
		self.old_E = [float(initial_conditions['E'])]
		# Infected (unquarantined)
		self.I = [float(initial_conditions['I'])]
		self.old_I = [float(initial_conditions['I'])]

		# Contacts
		self.total_contacts = []
		self.IR = []

		# The initial population
		self.N0 = self.S[0] + self.E[0] + self.I[0]+ float(initial_conditions['R']) + float(initial_conditions['Rq'])


	def initialize_bounds(self, alphas):
		# These are the bounds for S initially
		self.S_L = [self.S[0]]
		self.S_U = [self.S[0]]

		# These are the bounds for IR initially
		summ_contacts = 0
		for n,g in self.all_groups.items():
			new_contacts = n_contacts(self, g, alphas, self.mixing_method)
			summ_contacts += new_contacts*g.I[0]/g.N0
		self.IR_L = [summ_contacts]
		self.IR_U = [summ_contacts]


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


	# Attach other groups to make it easier to find variables of other groups
	def attach_other_groups(self,all_groups):
		self.all_groups = all_groups

	# Advances one time step, given the m_tests and a_tests variable
	def take_time_step(self, m_tests, a_tests):
		self.update_S(m_tests, a_tests)
		self.update_E(m_tests, a_tests)
		self.update_I(m_tests, a_tests)

		self.t += 1

	# Updates S
	def update_S(self, m_tests, a_tests):
		delta_S = -self.parameters['beta']*self.total_contacts[self.t]
		new_S = self.S[0] - self.parameters['beta']*sum([self.total_contacts[k] for k in range(0,self.t+1)])
		self.S += [new_S]
		self.old_S += [self.old_S[self.t]+delta_S]

	# Updates Exposed
	def update_E(self, m_tests, a_tests):
		delta_E = self.parameters['beta']*self.total_contacts[self.t] - self.parameters['sigma']*self.E[self.t]
		new_E = (
			(1-self.parameters['sigma'])**(self.t+1)*self.E[0] + 
			self.parameters['beta']*sum([(1-self.parameters['sigma'])**(self.t+1-k-1)*self.total_contacts[k] for k in range(0,self.t+1)])
		)
		self.E += [new_E]
		self.old_E += [self.old_E[self.t]+delta_E]


	# Updates infected
	def update_I(self, m_tests, a_tests):
		delta_I = self.parameters['sigma']*self.E[self.t] - self.parameters['mu']*self.I[self.t] - m_tests
		new_I = (
			(1-self.parameters['mu'])**(self.t+1)*self.I[0]+
			sum([(1-self.parameters['mu'])**(self.t+1-tao-1)*self.parameters['sigma']*(1-self.parameters['sigma'])**tao*self.E[0] for tao in range(0,self.t+1)])+
			sum([(1-self.parameters['mu'])**(self.t+1-tao-1)*self.parameters['sigma']*self.parameters['beta']*
				sum([(1-self.parameters['sigma'])**(tao-k-1)*self.total_contacts[k] for k in range(0,tao)])
			for tao in range(0,self.t+1)])-
			sum([(1-self.parameters['mu'])**(self.t+1-tao-1)*m_tests for tao in range(0,self.t+1)])
		)
		self.I += [new_I]
		self.old_I += [self.old_I[self.t]+delta_I*self.dt]




