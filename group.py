
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
		# Exposed (unquarentined)
		self.E = [float(parameters['initial-conditions']['E'])]
		# Infected (unquarentined)
		self.I = [float(parameters['initial-conditions']['I'])]
		# Unquarentined patients 
		self.N = [self.S + self.E + self.I]
		# Recovered (unquarentined)
		self.R = [float(parameters['initial-conditions']['R'])]

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

	




