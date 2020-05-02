from gurobipy import *


class DynamicalModel:
	def __init__(self, parameters, dt):

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
	def simulate(self, time_steps, m_tests_vec, a_tests_vec):

		for t in range(time_steps):
			for n in self.groups:
				self.groups[n].take_time_step(m_tests_vec[self.t], a_tests_vec[self.t])
			self.t +=1

	# Simulates the dynamics of the upper bound
	def simulate_upper(self, time_steps, m_tests_vec, a_tests_vec, rho, B_ICU, B_H):

		for t in range(time_steps):
			for n in self.groups:
				self.groups[n].take_time_step_upper(m_tests_vec[self.t], a_tests_vec[self.t], rho[self.t], B_ICU[self.t], B_H[self.t])
			self.t +=1

	def get_rho_bounds(self, time_steps):
		rho_lb_vector = []
		rho_ub_vector = []
		for t in range(time_steps):
			rho_lb_vector.append(0.0)
			rho_ub_vector.append(float('inf'))
		return (rho_lb_vector,rho_ub_vector)

	# Calculates Upper Bound
	def upper_bound(self, time_steps, A_tests_capacity, M_tests_capacity):
		# Create gurobi model
		M = Model()

		# Add variables to model
		B_ICU_vector = []
		B_H_vector = []
		A_tests_vector = []
		M_tests_vector = []
		rho_vector = []
		for t in range(time_steps):
			B_ICU_vector.append(M.addVar(vtype=GRB.CONTINUOUS, name="B_ICU_"+str(self.t)))
			B_H_vector.append(M.addVar(vtype=GRB.CONTINUOUS, name="B_H_"+str(self.t)))
			A_tests_vector.append(M.addVar(lb=0.0,vtype=GRB.CONTINUOUS, name="A_tests_"+str(self.t)))
			M_tests_vector.append(M.addVar(lb=0.0,vtype=GRB.CONTINUOUS, name="M_tests_"+str(self.t)))
			rho_vector.append(M.addVar(lb=0.0,vtype=GRB.CONTINUOUS, name="rho_"+str(self.t)))
		M.update()

		# Run simulation
		self.simulate_upper(time_steps, M_tests_vector, A_tests_vector, rho_vector, B_ICU_vector, B_H_vector)

		# Get bounds on rho
		rho_lb_vector,rho_ub_vector = self.get_rho_bounds(time_steps)


		# Add constraints
		for t in range(time_steps):
			# Molecular tests capacity
			M.addConstr(
				M_tests_vector[t] + 
				self.parameters["mu"]*(self.parameters["p_H"]+self.parameters["p_ICU"])*self.I[t]+
				self.parameters["mu"]*(self.Ia[t]+self.Ips[t]+self.Ims[t])+
				self.parameters['lambda_H']*self.H[t] +
				self.parameters['lambda_ICU']*self.ICU[t] 
				<= M_tests_capacity[t]
			)
			# Atomic tests capacity
			M.addConstr(A_tests_vector[t] <= M_tests_capacity[t])
			# TODO Constraints on B

			# Constraints on rho
			M.addConstr(rho_vector[t]<=rho_ub_vector[t])
			M.addConstr(rho_vector[t]>=rho_lb_vector[t])




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
		self.dt = 1


	def initialize_vars(self, initial_conditions):
		# Susceptible 
		self.S = [float(initial_conditions['S'])]
		# Exposed (unquarantined)
		self.E = [float(initial_conditions['E'])]
		# Infected (unquarantined)
		self.I = [float(initial_conditions['I'])]
		# Unquarantined patients 
		self.N = [self.S[0] + self.E[0] + self.I[0]]
		# Recovered (unquarantined)
		self.R = [float(initial_conditions['R'])]

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
	def take_time_step(self, m_tests, a_tests):
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
		self.update_H(m_tests, a_tests)
		self.update_ICU(m_tests, a_tests)
		self.update_D(m_tests, a_tests)

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
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]+self.parameters["p_ICU"])*self.I[self.t] - a_tests*self.R[self.t]/self.N[self.t]
		self.R += [self.R[self.t]+delta_R*self.dt]

	def update_R_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]+self.parameters["p_ICU"])*self.I[self.t] - a_tests
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
			self.parameters['lambda_H']*self.H[self.t] +
			self.parameters['lambda_ICU']*self.ICU[self.t] +
			a_tests*self.R[self.t]/self.N[self.t]
		)
		self.Rq += [self.Rq[self.t]+delta_Rq*self.dt]

	def update_Rq_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_Rq = (
			self.parameters['mu']*(self.Ia[self.t]+self.Ips[self.t]+self.Ims[self.t]) + 
			self.parameters['lambda_H']*self.H[self.t] +
			self.parameters['lambda_ICU']*self.ICU[self.t] +
			a_tests
		)
		self.Rq += [self.Rq[self.t]+delta_Rq*self.dt]




	def update_H(self, m_tests, a_tests):
		# For each group, calculate the entering amount
		entering_h = {}
		summ_entering_h = 0
		for n,g in self.all_groups.iteritems():
			entering_h[n] = self.parameters['mu']*self.parameters['p_H']*(g.I[self.t]+g.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
			summ_entering_h += entering_h[n]
		# Calculate number of beds
		beds = self.parameters['C_H']
		for n,g in self.all_groups.iteritems():
			beds-=g.H[self.t]

		delta_H = entering_h[self.name]*(1-(summ_entering_h-beds if summ_entering_h-beds>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6))
		self.H += [self.H[self.t]+delta_H*self.dt]

	def update_H_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_H = self.parameters['mu']*self.parameters['p_H']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU'])) - B_H
		self.H += [self.H[self.t]+delta_H*self.dt]



	def update_ICU(self, m_tests, a_tests):
		# For each group, calculate the entering amount
		entering_icu = {}
		summ_entering_icu = 0
		for n,g in self.all_groups.iteritems():
			entering_icu[n] = self.parameters['mu']*self.parameters['p_ICU']*(g.I[self.t]+g.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
			summ_entering_icu += entering_icu[n]
		# Calculate number of ICUs
		icus = self.parameters['C_ICU']
		for n,g in self.all_groups.iteritems():
			icus-=g.H[self.t]

		delta_ICU = entering_icu[self.name]*(1-(summ_entering_icu-icus if summ_entering_icu-icus>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6))
		self.ICU += [self.ICU[self.t]+delta_ICU*self.dt]

	def update_ICU_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		delta_ICU = self.parameters['mu']*self.parameters['p_ICU']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU'])) - B_ICU
		self.ICU += [self.ICU[self.t]+delta_ICU*self.dt]


	def update_D(self, m_tests, a_tests):
		# For each group, calculate the entering amount
		entering_h = {}
		summ_entering_h = 0
		for n,g in self.all_groups.iteritems():
			entering_h[n] = self.parameters['mu']*self.parameters['p_H']*(g.I[self.t]+g.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
			summ_entering_h += entering_h[n]
		# Calculate number of beds
		beds = self.parameters['C_H']
		for n,g in self.all_groups.iteritems():
			beds-=g.H[self.t]

		entering_icu = {}
		summ_entering_icu = 0
		for n,g in self.all_groups.iteritems():
			entering_icu[n] = self.parameters['mu']*self.parameters['p_ICU']*(g.I[self.t]+g.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
			summ_entering_icu += entering_icu[n]
		# Calculate number of ICUs
		icus = self.parameters['C_ICU']
		for n,g in self.all_groups.iteritems():
			icus-=g.H[self.t]

		delta_D = (
			entering_icu[self.name]*((summ_entering_icu-icus if summ_entering_icu-icus>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6))
			+ entering_h[self.name]*((summ_entering_h-beds if summ_entering_h-beds>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6))
			+ self.parameters['q_H'] * self.H[self.t]
			+ self.parameters['q_ICU'] * self.ICU[self.t]
		)

		self.D += [self.D[self.t]+delta_D*self.dt]

	def update_D_upper(self, m_tests, a_tests, rho, B_ICU, B_H):
		# For each group, calculate the entering amount

		delta_D = (
			B_H + B_ICU +
			+ self.parameters['q_H'] * self.H[self.t]
			+ self.parameters['q_ICU'] * self.ICU[self.t]
		)

		self.D += [self.D[self.t]+delta_D*self.dt]










