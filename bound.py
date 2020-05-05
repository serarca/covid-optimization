class Bounds:
	def __init__(self, parameters, dt):

		self.t = 0
		self.dt = dt
		self.rho_upper = float("inf")
		self.rho_lower = 0

		# Create groups from parameters
		self.groups = {}
		for n in parameters['seir-groups']:
			self.groups[n] = Bounds_SEIR_group(parameters['seir-groups'][n], parameters['global-parameters'], self.dt)

		# Attach other groups to each group
		for n in self.groups:
			self.groups[n].attach_other_groups(self.groups)

	# Obtains bounds for rho. Runs the simulation with bounded
	# states to obtain rho_lower and rho_upper
	def obtainBounds(self, time_steps):

		for t in range(time_steps):
			for n in self.groups:
				self.groups[n].take_time_step(self)
			self.t +=1


class Bounds_SEIR_group:
	# Time step
	def __init__(self, group_parameters,
				 global_parameters, dt):
		# Group name
		self.name = group_parameters['name']
		self.parameters = group_parameters['parameters']
		self.contacts = group_parameters['contacts']
		self.initial_conditions = group_parameters['initial-conditions']
		self.testingLimitsA = global_parameters['A_tests']
		self.testingLimitsM = global_parameters['M_tests']
		self.initialize_vars(self.initial_conditions)
		self.initialize_rhos()
		# Time step
		self.t = 0
		self.dt = 1


	def initialize_vars(self, initial_conditions):
		# Susceptible
		self.S_upper = [float(initial_conditions['S'])]
		# Exposed (unquarantined)
		self.E_upper = [float(initial_conditions['E'])]
		# Infected (unquarantined)
		self.I_upper = [float(initial_conditions['I'])]
		# Unquarantined patients
		self.N_upper = [self.S_upper[0]
						+ self.E_upper[0]
						+ self.I_upper[0]]


		# Susceptible Lower bound
		self.S_lower = [float(initial_conditions['S'])]
		# Exposed (unquarantined) Lower bound
		self.E_lower = [float(initial_conditions['E'])]
		# Infected (unquarantined) lower bound
		self.I_lower = [float(initial_conditions['I'])]
		# Unquarantined patients lower bound
		self.N_lower = [self.S_lower[0]
						+ self.E_lower[0]
						+ self.I_lower[0]]

	def initialize_rhos(self):
		# rho ~ (Sg sum_h(c_{gh} (Ih/Nh))
		self.rho_upper = [self.S_upper[0]
		 				  * sum([self.contacts[name]
						  		 * group.I_upper[0]
								 / group.N_upper[0]
								 for name,group in self.all_groups.iteritems()])]

		# rho ~ (Sg sum_h(c_{gh} (Ih/Nh)) lower bound
		self.rho_lower = [self.S_lower[0]
		 				  * sum([self.contacts[name]
						  		 * group.I_lower[0]
								 / group.N_lower[0]
								 for name,group in self.all_groups.iteritems()])]



	# Attach other groups to make it easier to find variables of other groups
	def attach_other_groups(self,all_groups):
		self.all_groups = all_groups

	# Advances one time step, given the m_tests and a_tests variable
	def take_time_step(self):
		self.update_N_lower()
		self.update_S_lower()
		self.update_E_lower()
		self.update_I_lower()

		self.update_N_lower()
		self.update_S_lower()
		self.update_E_lower()
		self.update_I_lower()

		self.update_rho_lower()
		self.update_rho_upper()

		self.t += 1


	# Updates N
	def update_N_lower(self):
		delta_N_lower = (
			- self.testingLimitsM
			- self.testingLimitsA
			- self.parameters['mu']*(self.parameters['p_H'] + self.parameters['p_ICU'])*self.I_upper[self.t]
		)
		self.N_lower += [self.N_lower[self.t]
						 + delta_N_lower * self.dt]

	# Updates S
	def update_S_lower(self):
		delta_S_lower = (-self.parameters['beta']
				   		 * self.rho_upper(self.t))
		self.S_lower += [self.S_lower[self.t]
						 + delta_S_lower * self.dt]

	def update_E_lower(self):
		delta_E_lower = (self.parameters['beta']
						 * self.rho_lower[self.t]
						 - self.parameters['sigma']
						 * self.E_lower[self.t])
		self.E_lower += [self.E_lower[self.t]
						 + delta_E_lower * self.dt]

	def update_I_lower(self):
		delta_I_lower =(self.parameters['sigma']
						* self.E_lower[self.t]
						- self.parameters['mu']
						* self.I_lower[self.t]
						- self.testingLimitsM)
		self.I_lower += [self.I_lower[self.t]
						 + delta_I_lower * self.dt]

	# To define the lower bound on rho, we will use that the
	# fraction that defines the probability of contact with an
	# infected decreases when you increase N. Hence, in order
	# to minimize rho, you would conduct no Antigen tests,
	# because these reduce N without changing I. Therefore,
	# you need only to look at different values of M tests for
	# each group. In this first aprox, we simply sample in the
	# simplex and keep the lowest rho we find.
	def update_rho_lower(self):
		sample_M_tests = generateMTests(k, self.all_groups, self.testingLimitsM)
		min_infected_contacts = float("inf")
		for M_tests in sample_M_tests:
			infected_contacts_lower = 0
			for name,group in self.all_groups.iteritems():
				infected_contacts_lower += (self.contacts[name]
									* (group.I_lower[self.t] * (1 - group.parameters['mu']) + group.parameters['sigma'] * group.E_lower[self.t] - M_tests_vector[group.name])
									/(group.N_upper[self.t] - M_tests_vector[group.name] - group.parameters['mu'] * (group.parameters['p_H'] + group.parameters['p_ICU']) * group.I_lower[self.t]))
			min_infected_contacts = min(min_infected_contacts, infected_contacts_lower)
		return self.S_lower[self.t + 1] * min_infected_contacts

	def compositions(t,s):
    q = [0] * t
    r = 0
    q[0] = s
    while True:
        yield q
        if q[0] == 0:
            if r==t-1:
                break
            else:
                q[0] = q[r] - 1
                q[r] = 0
                r = r + 1
        else:
            q[0] = q[0] - 1
            r = 1
        q[r] = q[r] + 1


	def generateMTests(k, all_groups, testingLimitsM):
		arr = []
		for c in compositions(len(all_groups), testingLimitsM):
			arr.append(list(c))
		sample_arr = random.sample(arr, k)
		sample_M_tests = []
		for random_tests in sample_arr:
			M_tests = {}
			for group in all_groups:
				M_tests[group.name] = random_tests.pop()
			sample_M_tests.append(M_tests)
		return sample_M_tests



	# Updates N
	def update_N_upper(self):
		delta_N_upper = (- self.parameters['mu']
						 * (self.parameters['p_H']
						 	+ self.parameters['p_ICU'])
						 * self.I_lower[self.t])
		self.N_upper += [self.N_upper[self.t]
						 + delta_N_upper * self.dt]

	# Updates S
	def update_S_upper(self):
		delta_S_upper = (-self.parameters['beta']
				   		 * self.rho_lower(self.t))
		self.S_upper += [self.S_upper[self.t]
						 + delta_S_upper * self.dt]

	def update_E_upper(self):
		delta_E_upper = (self.parameters['beta']
						 * self.rho_upper[self.t]
						 - self.parameters['sigma']
						 * self.E_upper[self.t])
		self.E_upper += [self.E_upper[self.t]
						 + delta_E_upper * self.dt]

	def update_I_upper(self):
		delta_I_upper =(self.parameters['sigma']
						* self.E_upper[self.t]
						- self.parameters['mu']
						* self.I_upper[self.t])
		self.I_upper += [self.I_upper[self.t]
						 + delta_I_upper * self.dt]

	# To define the upper bound on rho, we will use that the
	# fraction that defines the probability of contact with an
	# infected is decreasing in the number of M tests. Hence,
	# in order to maximize the number of infection, you'd not
	# do any molecular tests. Thus, the function to optimize
	# becomes a function only of the A tests (which reduce N).
	def update_rho_upper(self):
		sample_A_tests = generateATests(k, self.all_groups, self.testingLimitsA)
		min_infected_contacts = float("inf")
		for A_tests in sample_A_tests:
			infected_contacts_lower = 0
			for name,group in self.all_groups.iteritems():
				infected_contacts_upper += (self.contacts[name]
									* (group.I_upper[self.t] * (1 - group.parameters['mu']) + group.parameters['sigma'] * group.E_upper[self.t])
									/(group.N_lower[self.t] - A_tests[group.name] - group.parameters['mu'] * (group.parameters['p_H'] + group.parameters['p_ICU']) * group.I_upper[self.t]))
			max_infected_contacts = max(max_infected_contacts, infected_contacts_upper)
		return self.S_upper[self.t + 1] * max_infected_contacts

	def generateATests(k, all_groups, testingLimitsA):
		arr = []
		for c in compositions(len(all_groups), testingLimitsA):
			arr.append(list(c))
		sample_arr = random.sample(arr, k)
		sample_M_tests = []
		for random_tests in sample_arr:
			M_tests = {}
			for group in all_groups:
				M_tests[group.name] = random_tests.pop()
			sample_A_tests.append(M_tests)
		return sample_A_tests
