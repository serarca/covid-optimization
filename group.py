
class SEIR_group:
	# Time step
	dt = 1
	def __init__(self, parameters):
		# Group name
		self.name = parameters['name']

		# Time step
		self.t = 0.0

		# Susceptible 
		self.S = parameters['initial-conditions']['S']
		# Exposed (unquarentined)
		self.E = parameters['initial-conditions']['E']
		# Infected (unquarentined)
		self.I = parameters['initial-conditions']['I']
		# Unquarentined patients 
		self.N = self.S + self.E + self.I

		# Recovered (unquarentined)
		self.R = parameters['initial-conditions']['R']

	def attach_other_groups(self,all_groups):
		self.other_groups = all_groups
		

	def take_time_step(self, m_tests, a_tests):
		new_N = self.updated_N(m_tests, a_tests)

		self.N =  new_N

	def updated_N(self, m_tests, a_tests):
		new_N = self.N - m_tests*self.I/self.N - a_tests*self.R/self.N

