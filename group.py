from gurobipy import *
from collections import defaultdict


class DynamicalModel:
	def __init__(self, parameters, dt):
		self.parameters = parameters
		self.t = 0
		self.dt = dt

		# Create groups from parameters
		self.groups = {}
		for n in parameters['seir-groups']:
			self.groups[n] = SEIR_group(parameters['seir-groups'][n], self.dt)

		# Attach other groups to each group
		for n in self.groups:
			self.groups[n].attach_other_groups(self.groups)

	# Simulates the dynamics given a vector of molecular and atomic tests
	def simulate(self, time_steps, m_tests_vec, a_tests_vec, h_cap_vec, icu_cap_vec):
		for t in range(time_steps):
			for n in self.groups:
				self.groups[n].take_time_step(m_tests_vec[self.t], a_tests_vec[self.t], h_cap_vec[self.t], icu_cap_vec[self.t])
			self.t +=1

	# Simulates the dynamics of the upper bound
	def simulate_upper(self, time_steps, m_tests_vec, a_tests_vec, rho, B_ICU, B_H):

		for t in range(time_steps):
			for n in self.groups:
				self.groups[n].take_time_step_upper(m_tests_vec[n][self.t], a_tests_vec[n][self.t], rho[n][self.t], B_ICU[n][self.t], B_H[n][self.t])
			self.t +=1

	def get_rho_bounds(self, time_steps):
		boundsModel = Bounds(self.parameters, self.dt)
		boundsModel.obtainBounds(self.time_steps)

		rho_lb_vector = defaultdict(list)
		rho_ub_vector = defaultdict(list)
		for n in self.groups:
			rho_lb_vector[n] = boundsModel.groups[n].rho_lower
			rho_ub_vector[n] = boundsModel.groups[n].rho_upper
		return (rho_lb_vector, rho_ub_vector)

	# Calculates Upper Bound
	def upper_bound(self, time_steps, A_tests_capacity, M_tests_capacity, h_cap_vec, icu_cap_vec):
		# Create gurobi model
		M = Model()

		# Add variables to model
		B_ICU_vector = defaultdict(list)
		B_H_vector = defaultdict(list)
		A_tests_vector = defaultdict(list)
		M_tests_vector = defaultdict(list)
		rho_vector = defaultdict(list)

		for n in self.groups:
			for t in range(time_steps):
				B_ICU_vector[n].append(M.addVar(vtype=GRB.CONTINUOUS, name="B_ICU_"+str(self.t)))
				B_H_vector[n].append(M.addVar(vtype=GRB.CONTINUOUS, name="B_H_"+str(self.t)))
				A_tests_vector[n].append(M.addVar(vtype=GRB.CONTINUOUS, name="A_tests_"+str(self.t)))
				M_tests_vector[n].append(M.addVar(vtype=GRB.CONTINUOUS, name="M_tests_"+str(self.t)))
				rho_vector[n].append(M.addVar(vtype=GRB.CONTINUOUS, name="rho_"+str(self.t)))
		M.update()

		print("Created variables")

		# Run simulation
		self.simulate_upper(time_steps, M_tests_vector, A_tests_vector, rho_vector, B_ICU_vector, B_H_vector)

		print("Ran simulation")

		# Add constraints on rho
		rho_lb_vector,rho_ub_vector = self.get_rho_bounds(time_steps)
		for n in self.groups:
			for t in range(time_steps):
				M.addConstr(rho_vector[n][t]<=rho_ub_vector[n][t])
				M.addConstr(rho_vector[n][t]>=rho_lb_vector[n][t])


		# Add nonnegativity constraints on the molecular tests
		for n in self.groups:
			for t in range(time_steps):
				# Nonnegativity
				M.addConstr(M_tests_vector[n][t] >= 0)

		# Calculate applied molecular tests apart from the randomized testing
		self.applied_M_tests = defaultdict(list)
		for n in self.groups:
			for t in range(time_steps):
				self.applied_M_tests[n].append(
					self.groups[n].parameters['mu']*(self.groups[n].parameters['p_H']+self.groups[n].parameters['p_ICU'])*self.groups[n].I[t]
					+ self.groups[n].parameters['mu']*(self.groups[n].Ia[t]+self.groups[n].Ips[t]+self.groups[n].Ims[t])
					+ self.groups[n].parameters['lambda_H_R']*self.groups[n].H[t]
					+ self.groups[n].parameters['lambda_ICU_R']*self.groups[n].ICU[t]
				)
		# Add upper bound on molecular tests
		for t in range(time_steps):
			M.addConstr(
				quicksum(M_tests_vector[n][t] for n in self.groups)
				+ quicksum(self.applied_M_tests[n][t] for n in self.groups)
				<= M_tests_capacity[t]
			)

		# Add bounds on antibody tests
		for n in self.groups:
			for t in range(time_steps):
				# Nonnegativity
				M.addConstr(A_tests_vector[n][t] >= 0)
		# Upper bounds
		for t in range(time_steps):
			M.addConstr(
				quicksum(A_tests_vector[n][t] for n in self.groups)
				<= A_tests_capacity[t]
			)

		# Add upper bounds on B_ICU and B_H
		for n in self.groups:
			for t in range(time_steps):
				M.addConstr(B_H_vector[n][t] <= self.groups[n].flow_H(t))
				M.addConstr(B_ICU_vector[n][t] <= self.groups[n].flow_ICU(t))

		# Add lower bounds on B_ICU and B_H
		for t in range(time_steps):
			M.addConstr(
				quicksum(self.groups[n].flow_H(t) - B_H_vector[n][t] for n in self.groups)
				<= h_cap_vec[t]
				- quicksum((1-self.groups[n].parameters['lambda_H_R']-self.groups[n].parameters['lambda_H_D'])*self.groups[n].H[t] for n in self.groups)
			)
			M.addConstr(
				quicksum(self.groups[n].flow_ICU(t) - B_ICU_vector[n][t] for n in self.groups)
				<= icu_cap_vec[t]
				- quicksum((1-self.groups[n].parameters['lambda_ICU_R']-self.groups[n].parameters['lambda_ICU_D'])*self.groups[n].ICU[t] for n in self.groups)
			)

		M.update()



class SEIR_group:
	# Time step
	def __init__(self, group_parameters, dt):
		# Group name
		self.name = group_parameters['name']
		self.parameters = group_parameters['parameters']
		self.contacts = group_parameters['contacts']
		self.initial_conditions = group_parameters['initial-conditions']
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

	# Advances one time step, given the m_tests and a_tests variable
	def take_time_step_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		self.update_N_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_S_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_E_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_I_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_R_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_Ia_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_Ips_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_Ims_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_Iss_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_Rq_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_H_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_ICU_upper(m_tests, a_tests, rho, B_ICU, B_H)
		self.update_D_upper(m_tests, a_tests, rho, B_ICU, B_H)

		self.t += 1

	# Gives flow of how many people flowing to H
	def flow_H(self, t):
		return self.parameters['mu']*self.parameters['p_H']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))

	# Gives flow of how many people flowing to ICU
	def flow_ICU(self, t):
		return self.parameters['mu']*self.parameters['p_ICU']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))


	# Updates N
	def update_N(self, m_tests, a_tests):
		delta_N = (
			- m_tests*self.I[self.t]/self.N[self.t]
			- a_tests*self.R[self.t]/self.N[self.t]
			- self.parameters['mu']*(self.parameters['p_H'] + self.parameters['p_ICU'])*self.I[self.t]
		)
		self.N += [self.N[self.t]+delta_N*self.dt]

	def update_N_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_N = (
			- m_tests
			- a_tests
			- self.parameters['mu']*(self.parameters['p_H'] + self.parameters['p_ICU'])*self.I[self.t]
		)
		self.N += [self.N[self.t]+delta_N*self.dt]

	# Updates S
	def update_S(self, m_tests, a_tests):
		infections = 0
		for name,group in self.all_groups.iteritems():
			infections += self.contacts[name]*group.I[self.t]/group.N[self.t]

		delta_S = -self.parameters['beta']*self.S[self.t]*infections
		self.S += [self.S[self.t]+delta_S*self.dt]

	def update_S_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_S = -self.parameters['beta'] * rho
		self.S += [self.S[self.t]+delta_S*self.dt]


	# Updates Exposed
	def update_E(self, m_tests, a_tests):
		infections = 0
		for name,group in self.all_groups.iteritems():
			infections += self.contacts[name]*group.I[self.t]/group.N[self.t]
		delta_E = self.parameters['beta']*self.S[self.t]*infections - self.parameters['sigma']*self.E[self.t]
		self.E += [self.E[self.t]+delta_E*self.dt]

	def update_E_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_E = self.parameters['beta']*rho - self.parameters['sigma']*self.E[self.t]
		self.E += [self.E[self.t]+delta_E*self.dt]


	# Updates infected
	def update_I(self, m_tests, a_tests):
		delta_I = self.parameters['sigma']*self.E[self.t] - self.parameters['mu']*self.I[self.t] - m_tests*self.I[self.t]/self.N[self.t]
		self.I += [self.I[self.t]+delta_I*self.dt]

	def update_I_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_I = self.parameters['sigma']*self.E[self.t] - self.parameters['mu']*self.I[self.t] - m_tests
		self.I += [self.I[self.t]+delta_I*self.dt]


	# Updates recovered
	def update_R(self, m_tests, a_tests):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]-self.parameters["p_ICU"])*self.I[self.t] - a_tests*self.R[self.t]/self.N[self.t]
		self.R += [self.R[self.t]+delta_R*self.dt]


	def update_R_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]-self.parameters["p_ICU"])*self.I[self.t] - a_tests
		self.R += [self.R[self.t]+delta_R*self.dt]


	# Updates infected in quarantine
	def update_Ia(self, m_tests, a_tests):
		delta_Ia = self.parameters['p_Ia']*m_tests*self.I[self.t]/self.N[self.t] - self.parameters['mu']*self.Ia[self.t]
		self.Ia += [self.Ia[self.t]+delta_Ia*self.dt]

	def update_Ips(self, m_tests, a_tests):
		delta_Ips = self.parameters['p_Ips']*m_tests*self.I[self.t]/self.N[self.t] - self.parameters['mu']*self.Ips[self.t]
		self.Ips += [self.Ips[self.t]+delta_Ips*self.dt]

	def update_Ims(self, m_tests, a_tests):
		delta_Ims = self.parameters['p_Ims']*m_tests*self.I[self.t]/self.N[self.t] - self.parameters['mu']*self.Ims[self.t]
		self.Ims += [self.Ims[self.t]+delta_Ims*self.dt]

	def update_Iss(self, m_tests, a_tests):
		delta_Iss = self.parameters['p_Iss']*m_tests*self.I[self.t]/self.N[self.t] - self.parameters['mu']*self.Iss[self.t]
		self.Iss += [self.Iss[self.t]+delta_Iss*self.dt]

	def update_Ia_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_Ia = self.parameters['p_Ia']*m_tests - self.parameters['mu']*self.Ia[self.t]
		self.Ia += [self.Ia[self.t]+delta_Ia*self.dt]

	def update_Ips_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_Ips = self.parameters['p_Ips']*m_tests - self.parameters['mu']*self.Ips[self.t]
		self.Ips += [self.Ips[self.t]+delta_Ips*self.dt]

	def update_Ims_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_Ims = self.parameters['p_Ims']*m_tests - self.parameters['mu']*self.Ims[self.t]
		self.Ims += [self.Ims[self.t]+delta_Ims*self.dt]

	def update_Iss_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_Iss = self.parameters['p_Iss']*m_tests - self.parameters['mu']*self.Iss[self.t]
		self.Iss += [self.Iss[self.t]+delta_Iss*self.dt]


	# Update recovered in quarentine
	def update_Rq(self, m_tests, a_tests):
		delta_Rq = (
			self.parameters['mu']*(self.Ia[self.t]+self.Ips[self.t]+self.Ims[self.t]) +
			self.parameters['lambda_H_R']*self.H[self.t] +
			self.parameters['lambda_ICU_R']*self.ICU[self.t] +
			a_tests*self.R[self.t]/self.N[self.t]
		)
		self.Rq += [self.Rq[self.t]+delta_Rq*self.dt]


	def update_Rq_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
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
		for n,g in self.all_groups.iteritems():
			entering_h[n] = self.all_groups[n].flow_H(self.t)
			summ_entering_h += entering_h[n]
			summ_staying_h += (1-g.parameters['lambda_H_R']-g.parameters['lambda_H_D'])*g.H[self.t]

		delta_H = (
			- (self.parameters["lambda_H_R"] + self.parameters["lambda_H_D"])*self.H[self.t]
			+ entering_h[self.name]*(1-(summ_entering_h-h_cap+summ_staying_h if summ_entering_h-h_cap+summ_staying_h>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6))
		)
		self.H += [self.H[self.t]+delta_H*self.dt]

	def update_H_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_H = (
			- (self.parameters["lambda_H_R"] + self.parameters["lambda_H_D"])*self.H[self.t]
			+ self.parameters['mu']*self.parameters['p_H']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
			- B_H
		)
		self.H += [self.H[self.t]+delta_H*self.dt]



	def update_ICU(self, m_tests, a_tests, h_cap, icu_cap):
		# For each group, calculate the entering amount
		entering_icu = {}
		summ_entering_icu = 0
		summ_staying_icu = 0
		for n,g in self.all_groups.iteritems():
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

	def update_ICU_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_ICU = (
			- (self.parameters["lambda_ICU_R"] + self.parameters["lambda_ICU_D"])*self.ICU[self.t]
			+ self.parameters['mu']*self.parameters['p_ICU']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
			- B_ICU
		)
		self.ICU += [self.ICU[self.t]+delta_ICU*self.dt]


	def update_D(self, m_tests, a_tests, h_cap, icu_cap):
		# For each group, calculate the entering amount
		entering_h = {}
		summ_entering_h = 0
		summ_staying_h = 0
		for n,g in self.all_groups.iteritems():
			entering_h[n] = self.all_groups[n].flow_H(self.t)
			summ_entering_h += entering_h[n]
			summ_staying_h += (1-g.parameters['lambda_H_R']-g.parameters['lambda_H_D'])*g.H[self.t]

		entering_icu = {}
		summ_entering_icu = 0
		summ_staying_icu = 0
		for n,g in self.all_groups.iteritems():
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

	def update_D_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		# For each group, calculate the entering amount

		delta_D = (
			self.parameters["lambda_H_D"]*self.H[self.t]
			+ self.parameters["lambda_ICU_D"]*self.ICU[self.t]
			+ B_H
			+ B_ICU
		)

		self.D += [self.D[self.t]+delta_D*self.dt]
