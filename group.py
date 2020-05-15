from collections import defaultdict
from bound import Bounds

class DynamicalModel:
	def __init__(self, parameters, initialization, dt, time_steps):
		self.parameters = parameters
		self.t = 0
		self.dt = dt
		self.time_steps = time_steps

		# Create groups from parameters
		self.groups = {}
		for n in parameters['seir-groups']:
			self.groups[n] = SEIR_group(parameters['seir-groups'][n], initialization[n], self.dt)

		# Attach other groups to each group
		for n in self.groups:
			self.groups[n].attach_other_groups(self.groups)

		# Initialize total contacts
		for n in self.groups:
			self.groups[n].initialize_total_contacts()

		# Fix number of beds and icus
		self.beds = self.parameters['global-parameters']['C_H']
		self.icus = self.parameters['global-parameters']['C_ICU']

		# Initialize objective values
		self.economic_values = [float("nan")]
		self.rewards = [float("nan")]
		self.deaths = [float("nan")]

	def take_time_step(self, m_tests, a_tests, alphas):
		for n in self.groups:
			self.groups[n].take_time_step(m_tests[n], a_tests[n], alphas)
		for n in self.groups:
			# TODO check this
			self.groups[n].update_total_contacts(self.t+1)

		# Calculate economic values
		state = self.get_state(self.t+1)
		deaths = sum([group.D[self.t+1] for name,group in self.groups.iteritems()])
		deaths_value = sum([group.D[self.t+1]*group.parameters['economic_value']['death_value'] for name,group in self.groups.iteritems()])
		economic_value = get_economic_value(state, alphas)
		reward = economic_value - deaths_value
		result = {
			"state": state,
			"economic_value": get_economic_value(state, alphas),
			"deaths": deaths,
			"deaths_value": deaths_value,
			"reward":reward,
		}

		# Update economic values
		self.rewards.append(economic_value)
		self.deaths.append(deaths)
		self.rewards.append(reward)

		# Update time
		self.t += 1
		return result

	# Simulates the dynamics given a vector of molecular tests, atomic tests and alphas
	def simulate(self, m_tests_vec, a_tests_vec, alphas_vec):
		for t in range(self.time_steps):
			take_time_step(self, m_tests_vec[t], a_tests_vec[t], alphas_vec[t])

	# Given a state and set of alphas, returns the economic value
	def get_economic_value(self, state, alphas):
		value = 0
		for group in state:
			value += (
				self.groups[group].parameters['economic_value']['work_value']*(
					alphas[group]['work']+
					self.groups[group].parameters['economic_value']['lockdown_fraction']*(1-alphas[group]['work'])
				)
			)

	## TODO
	def get_state(self, t):


	def get_total_deaths(self):

	def get_total_economic_value(self):

	def get_total_reward(self):

	def print_stats(self):
		print("Economic Value: "+str(self.get_total_economic_value()))
		print("Deaths "+str(self.get_total_deaths()))

class SEIR_group:
	# Time step
	def __init__(self, group_parameters, group_initialization, dt):
		# Group name
		self.name = group_parameters['name']
		self.parameters = group_parameters['parameters']
		self.contacts = group_parameters['contacts']
		self.initial_conditions = group_initialization
		self.same_biomarkers = group_parameters['same_biomarkers']
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
		# Economic value
		self.Econ = [0.0]

		# Contacts
		self.total_contacts = []

	def initialize_total_contacts(self):
		self.update_total_contacts(0)

	## TODO: update this function
	def update_total_contacts(self, t):
		if (len(self.total_contacts) == t):
			summ_contacts = 0
			for n,g in self.all_groups.iteritems():
				biomarkers_pop = sum([self.all_groups[g_bio].N[self.t] + self.all_groups[g_bio].Rq[self.t] for g_bio in g.same_biomarkers])
				summ_contacts+=self.contacts[n]*g.I[t]/(biomarkers_pop if biomarkers_pop!=0 else 10e-6)				
			self.total_contacts.append(summ_contacts*self.S[t])
		else:
			assert(False)


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

	# Gives flow of how many people flowing to H
	def flow_H(self, t):
		if self.parameters['p_H'] != 0.0:
			return self.parameters['mu']*self.parameters['p_H']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
		else:
			return 0.0

	# Gives flow of how many people flowing to ICU
	def flow_ICU(self, t):
		if self.parameters['p_ICU'] != 0.0:
			return self.parameters['mu']*self.parameters['p_ICU']*(self.I[self.t]+self.Iss[self.t]/(self.parameters['p_H']+self.parameters['p_ICU']))
		else:
			return 0.0

	# Updates N
	def update_N(self, m_tests, a_tests):
		delta_N = (
			- m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
			- a_tests*self.R[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
			- self.parameters['mu']*(self.parameters['p_H'] + self.parameters['p_ICU'])*self.I[self.t]
		)
		self.N += [self.N[self.t]+delta_N*self.dt]

	# Updates S
	def update_S(self, m_tests, a_tests):
		## TODO might be wrong
		delta_S = -self.parameters['beta']*self.total_contacts[self.t-1]
		self.S += [self.S[self.t]+delta_S*self.dt]

	# Updates Exposed
	def update_E(self, m_tests, a_tests):
		delta_E = self.parameters['beta']*self.total_contacts[self.t-1] - self.parameters['sigma']*self.E[self.t]
		self.E += [self.E[self.t]+delta_E*self.dt]


	# Updates infected
	def update_I(self, m_tests, a_tests):
		delta_I = self.parameters['sigma']*self.E[self.t] - self.parameters['mu']*self.I[self.t] - m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
		self.I += [self.I[self.t]+delta_I*self.dt]


	# Updates recovered
	def update_R(self, m_tests, a_tests):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]-self.parameters["p_ICU"])*self.I[self.t] - a_tests*self.R[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
		self.R += [self.R[self.t]+delta_R*self.dt]


	# Updates infected in quarantine
	def update_Ia(self, m_tests, a_tests):
		delta_Ia = self.parameters['p_Ia']*m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6) - self.parameters['mu']*self.Ia[self.t]
		self.Ia += [self.Ia[self.t]+delta_Ia*self.dt]

	def update_Ips(self, m_tests, a_tests):
		delta_Ips = self.parameters['p_Ips']*m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6) - self.parameters['mu']*self.Ips[self.t]
		self.Ips += [self.Ips[self.t]+delta_Ips*self.dt]

	def update_Ims(self, m_tests, a_tests):
		delta_Ims = self.parameters['p_Ims']*m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6) - self.parameters['mu']*self.Ims[self.t]
		self.Ims += [self.Ims[self.t]+delta_Ims*self.dt]

	def update_Iss(self, m_tests, a_tests):
		delta_Iss = self.parameters['p_Iss']*m_tests*self.I[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6) - self.parameters['mu']*self.Iss[self.t]
		self.Iss += [self.Iss[self.t]+delta_Iss*self.dt]


	# Update recovered in quarentine
	def update_Rq(self, m_tests, a_tests):
		delta_Rq = (
			self.parameters['mu']*(self.Ia[self.t]+self.Ips[self.t]+self.Ims[self.t]) +
			self.parameters['lambda_H_R']*self.H[self.t] +
			self.parameters['lambda_ICU_R']*self.ICU[self.t] +
			a_tests*self.R[self.t]/(self.N[self.t] if self.N[self.t]!=0 else 10e-6)
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
