
class SEIR_group:
	# Time step
	dt = 1
	def __init__(self, parameters):
		# Group name
		self.name = parameters['name']
		self.parameters = parameters['parameters']

		# Time step
		self.t = 0

		# Susceptible 
		self.S = [float(parameters['initial-conditions']['S'])]
		# Exposed (unquarantined)
		self.E = [float(parameters['initial-conditions']['E'])]
		# Infected (unquarantined)
		self.I = [float(parameters['initial-conditions']['I'])]
		# Unquarantined patients 
		self.N = [self.S + self.E + self.I]
		# Recovered (unquarantined)
		self.R = [float(parameters['initial-conditions']['R'])]

		# Infected quarantined with different degrees of severity
		self.I_a = [float(parameters['initial-conditions']['I_a'])]
		self.I_ps = [float(parameters['initial-conditions']['I_ps'])]
		self.I_ms = [float(parameters['initial-conditions']['I_ms'])]
		self.I_ss = [float(parameters['initial-conditions']['I_ss'])]

		# Recovered quanrantined
		self.Rq = [float(parameters['initial-conditions']['Rq'])]

		# In hospital bed
		self.H = [float(parameters['initial-conditions']['H'])]
		# In ICU
		self.ICU = [float(parameters['initial-conditions']['ICU'])]
		# Dead
		self.D = [float(parameters['initial-conditions']['D'])]



	# Attach other groups to make it easier to find variables of other groups
	def attach_other_groups(self,all_groups):
		self.all_groups = all_groups

	# Advances one time step, given the m_tests and a_tests variable
	def take_time_step(self, m_tests, a_tests):
		self.update_N(m_tests, a_tests)

	# Updates N
	def update_N(self, m_tests, a_tests):
		delta_N = -m_tests*self.I[self.t]/self.N[self.t] - a_tests*self.R[self.t]/self.N[self.t]
		self.N += [self.N[self.t]+delta_N*dt]

	# Updates S
	def update_S(self, m_tests, a_tests):
		infections = 0
		for name,group in self.all_groups.iteritems():
			infections += self.parameters['contacts'][name]*group.I[self.t]/group.N[self.t]

		delta_S = -self.parameters['beta']*self.S[self.t]*infections
		self.S += [self.S[self.t]+delta_S*dt]

	def update_E(self, m_tests, a_tests):
		infections = 0
		for name,group in self.all_groups.iteritems():
			infections += self.parameters['contacts'][name]*group.I[self.t]/group.N[self.t]
		delta_E = self.parameters['beta']*self.S[self.t]*infections - self.parameters['sigma']*self.E[self.t]
		self.E += [self.E[self.t]+delta_E*dt]

	def update_I(self, m_tests, a_tests):
		delta_I = self.parameters['sigma']*self.E[self.t] - self.parameters['mu']*self.I[self.t] - m_tests*self.I[self.t]/self.N[self.t]
		self.I += [self.I[self.t]+delta_I*dt]

	def update_R(self, m_tests, a_tests):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]+self.parameters["p_ICU"])*self.I[self.t] - a_tests*self.R[self.t]/self.N[self.t]
		self.R += [self.R[self.t]+delta_R*dt]

	def update_Ia(self, m_tests, a_tests):
		delta_Ia = self.parameters['p_Ia']*m_tests*self.I[self.t]/self.N[self.t] - parameters['mu']*self.Ia[self.t]
		self.Ia += [self.Ia[self.t]+delta_Ia*dt]

	def update_Ips(self, m_tests, a_tests):
		delta_Ips = self.parameters['p_Ips']*m_tests*self.I[self.t]/self.N[self.t] - parameters['mu']*self.Ips[self.t]
		self.Ips += [self.Ips[self.t]+delta_Ips*dt]

	def update_Ims(self, m_tests, a_tests):
		delta_Ims = self.parameters['p_Ims']*m_tests*self.I[self.t]/self.N[self.t] - parameters['mu']*self.Ims[self.t]
		self.Ims += [self.Ims[self.t]+delta_Ims*dt]

	def update_Iss(self, m_tests, a_tests):
		delta_Iss = self.parameters['p_Iss']*m_tests*self.I[self.t]/self.N[self.t] - parameters['mu']*self.Iss[self.t]
		self.Iss += [self.Iss[self.t]+delta_Iss*dt]

	def update_Rq(self, m_tests, a_tests):
		delta_Rq = (
			self.parameters['mu']*(self.Ia[self.t]+self.Ips[self.t]+self.Ims[self.t]) + 
			self.parameters['lambda_H']*self.H[self.t] +
			self.parameters['lambda_ICU']*self.ICU[self.t] +
			a_tests*self.R[self.t]/self.N[self.t]
		)
		self.Rq += [self.Rq[self.t]+delta_Rq*dt]

	def update_H(self, m_tests, a_tests):
		# For each group, calculate the entering amount
		entering = {}
		summ_entering = 0
		for n,g in self.all_groups:
			entering[g] = parameters['mu']*parameters['p_H']*(g.I[self.t]+g.I_ss[self.t])
			summ_entering += entering[g]
		# Calculate number of beds
		beds = self.parameters['C_H']
		for n,g in self.all_groups:
			beds-=g.H[self.t]

		delta_H = entering[self.name]*(1-(summ_entering-beds if summ_entering-beds>0 else 0)/summ_entering)














