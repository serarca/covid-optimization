from collections import defaultdict
import numpy as np
import pandas as pd
import math
import gurobipy as gb
from copy import deepcopy

all_activities = ['home','leisure','other','school','transport','work']
END_DAYS = 14

full_open_policy = {
	"home": 1.0,
	"leisure": 1.0,
	"other": 1.0,
	"school": 1.0,
	"transport": 1.0,
	"work": 1.0
}	


def n_contacts(group_g, group_h, alphas, mixing_method):

	n = {"total":0}
	if mixing_method['name'] == "maxmin":
		for activity in alphas[group_g.name]:
			n["total"] += group_g.contacts[activity][group_h.name]*(
					(alphas[group_g.name][activity]*math.exp(alphas[group_g.name][activity]*mixing_method['param']) + alphas[group_h.name][activity]*math.exp(alphas[group_h.name][activity]*mixing_method['param']))
					/(math.exp(alphas[group_g.name][activity]*mixing_method['param'])+math.exp(alphas[group_h.name][activity]*mixing_method['param']))
				)
	elif mixing_method['name'] == "mult":
		for activity in alphas[group_g.name]:
			value = group_g.contacts[activity][group_h.name]*(alphas[group_g.name][activity]**mixing_method['param_alpha'])*(alphas[group_h.name][activity]**mixing_method['param_beta'])
			
			n[activity] = value
			n["total"] += value

			if alphas[group_h.name][activity] < 0:
				print(f"group_h.name: {group_h.name}")
				print(f"act: {activity}")
				print(n)
				print(f" group_g.contacts[activity][group_h.name]:{group_g.contacts[activity][group_h.name]}")
				print(f" alphas[group_g.name][activity]:{alphas[group_g.name][activity]}")
				print(f"alphas[group_h.name][activity] :{alphas[group_h.name][activity]}")
				print(f"mixing_method['param_alpha']: {mixing_method['param_alpha']}")
				print(f"mixing_method['param_beta']: {mixing_method['param_beta']}")
				print(f"alphas[group_g.name][activity]**mixing_method['param_alpha']: {alphas[group_g.name][activity]**mixing_method['param_alpha']}")
				print(f"alphas[group_h.name][activity]**mixing_method['param_beta']:{alphas[group_h.name][activity]**mixing_method['param_beta']}")

	elif mixing_method['name'] == "min":
		for activity in alphas[group_g.name]:
			n["total"] += group_g.contacts[activity][group_h.name]*min(alphas[group_g.name][activity],alphas[group_h.name][activity])
	elif mixing_method['name'] == "max":
		for activity in alphas[group_g.name]:
			n["total"] += group_g.contacts[activity][group_h.name]*max(alphas[group_g.name][activity],alphas[group_h.name][activity])
	else:
		assert(False)

	return n


class DynamicalModel:
	def __init__(self, parameters, econ_params, experiment_params, initialization, dt, time_steps, mixing_method, start_day, eta, transport_lb_work_fraction = 0 , extra_data = False):
		self.parameters = deepcopy(parameters)
		self.econ_params = deepcopy(econ_params)
		self.experiment_params = deepcopy(experiment_params)
		self.t = 0
		self.dt = dt
		self.END_DAYS = END_DAYS
		self.time_steps = time_steps + END_DAYS
		self.initialization = deepcopy(initialization)
		self.mixing_method = mixing_method
		self.extra_data = extra_data
		self.use_gurobi_vars = False
		self.lockdown_controls = []
		self.m_tests_controls = []
		self.a_tests_controls = []
		self.transport_lb_work_fraction = transport_lb_work_fraction
		self.start_day = start_day
		self.age_groups = list(self.parameters['seir-groups'].keys())

		# Modify betas
		for ag in self.parameters['seir-groups']:
			self.parameters['seir-groups'][ag]['parameters']['beta'] = self.parameters['seir-groups'][ag]['parameters']['beta'][self.start_day:self.start_day+self.time_steps]
			for k in range(END_DAYS):
				self.parameters['seir-groups'][ag]['parameters']['beta'][self.time_steps-k-1] = 0.0

		# Modify eta
		self.econ_params["employment_params"]["eta"] = eta
		self.econ_params["employment_params"]["nu"] = 1 - eta - self.econ_params["employment_params"]["gamma"]

		# Create groups from parameters
		self.groups = {}
		for n in parameters['seir-groups']:
			self.groups[n] = SEIR_group(self.parameters['seir-groups'][n], self.initialization[n], self.dt, self.mixing_method, self.time_steps, self)


		# Attach other groups to each group
		for n in self.groups:
			self.groups[n].attach_other_groups(self.groups)

		# Fix number of beds and icus
		self.beds = self.parameters['global-parameters']['C_H']
		self.icus = self.experiment_params['icus']

		# Initialize objective values
		self.economic_values = [float("nan")]
		self.rewards = [float("nan")]
		self.deaths = [float("nan")]

		# Initialize total population
		self.total_population = sum([sum([self.initialization[group][cat] for cat in self.initialization[group].keys()]) for group in self.initialization.keys()])

		# Initialize number of contacts
		if extra_data:
			self.n_contacts = [{g_name1:{g_name2:float('inf') for g_name2 in self.groups} for g_name1 in self.groups} for i in range(self.time_steps)]
			self.n_infections = [{g_name1:{g_name2:float('inf') for g_name2 in self.groups} for g_name1 in self.groups} for i in range(self.time_steps)]
			self.n_infections_act = [{act: {g_name1:{g_name2:float('inf') for g_name2 in self.groups} for g_name1 in self.groups} for act in all_activities} for i in range(self.time_steps)]

	def take_end_steps(self):

		all_m_tests = []
		all_a_tests = []
		all_alphas = []
		for t in range(END_DAYS):
			m_tests = {ag:0 for ag in self.age_groups}
			a_tests = {ag:0 for ag in self.age_groups}
			alphas = {ag:deepcopy(full_open_policy) for ag in self.age_groups}
			self.take_time_step(m_tests, a_tests, alphas)
			all_m_tests.append(m_tests)
			all_a_tests.append(a_tests)
			all_alphas.append(alphas)
		assert(self.t == self.time_steps)

		return (all_alphas,all_a_tests,all_m_tests)


	def take_time_step(self, m_tests, a_tests, alphas, B_H = False, B_ICU = False, B_ICU_perc = False):
		# store time when current group is being updated
		time_of_flow = self.t
		
		for n in self.groups:
			self.groups[n].update_total_contacts(time_of_flow, alphas)

			if (B_H is not False) and (B_ICU is not False):
				if(isinstance(B_H[n], gb.Var) or isinstance(B_ICU[n], gb.Var) ):
					print("Take Time Step: B_H and B_ICU are gurobi variables. Skipping bounding and capcity assertion checks.")
					self.use_gurobi_vars = True

		# Store Lockdown and testing controls for time t
		self.lockdown_controls += [alphas]
		self.m_tests_controls += [m_tests]
		self.a_tests_controls += [a_tests]

		if (B_H is not False) and (B_ICU is not False) and (self.use_gurobi_vars is not True):
			#if ()
			B_H, B_ICU = self.cap_bounce_variables(B_H, B_ICU)

			for n in self.groups:
				self.groups[n].take_time_step(m_tests[n], a_tests[n], self.beds, self.icus, B_H[n], B_ICU[n])
		elif (B_ICU_perc is not False):
			for n in self.groups:
				self.groups[n].take_time_step(m_tests[n], a_tests[n], self.beds, self.icus, False, False, B_ICU_perc = B_ICU_perc[n])
		else:
			# print('group.py(). Taking a step at t={}.'.format(time_of_flow))
			# print('Total in ICU: {}'.format(sum(self.groups[n].ICU[time_of_flow] for n in self.groups)))
			for n in self.groups:
				self.groups[n].take_time_step(m_tests[n], a_tests[n], self.beds, self.icus, False, False)

		# Calculate economic values
		state = self.get_state(self.t+1)

		#print("People at the ICU in the next state: {}".format(sum(state[n]["ICU"] for n in self.groups)))
		#print("Total beds at the ICU: {}".format(self.icus))
		if( self.use_gurobi_vars is not True ):
			assert self.icus * self.dt - sum(state[n]["ICU"] for n in self.groups) > - 1e-10, f'Total ICU: {sum(state[n]["ICU"] for n in self.groups)}, total beds: {self.dt * self.icus}'


		deaths = sum([group.D[self.t+1]-group.D[self.t] for name,group in self.groups.items()])
		deaths_value = sum([(group.D[self.t+1]-group.D[self.t])*(self.econ_params["econ_cost_death"][name]+self.experiment_params["xi"]) for name,group in self.groups.items()])
		economic_value = self.get_economic_value(state, alphas)
		reward = economic_value - deaths_value
		result = {
			"state": state,
			"economic_value": economic_value,
			"deaths": deaths,
			"deaths_value": deaths_value,
			"reward":reward,
		}

		# Update economic values
		self.economic_values.append(economic_value)
		self.deaths.append(deaths)
		self.rewards.append(reward)

		# Update time
		self.t += 1
		return result

	# Cap bounce variables to ensure feasibility
	def cap_bounce_variables(self, B_H, B_ICU):
		tol = 1e-4  # bounce a bit more, to make sure capacities are met
		time_of_flow = self.t # use dynModel t as clock

		# Cap bounces at no more than the level of flow_H / flow_ICU
		for n,g in self.groups.items():

			if (B_H[n] > self.dt * g.flow_H()):
				# print('WARNING.group.py() Capping B_H for group {} at time {}'.format(n,time_of_flow))
				B_H[n] = self.dt * g.flow_H()

			if (B_ICU[n] > self.dt * g.flow_ICU()):
				# print('WARNING. group.py() Capping B_ICU for group {} at time {}'.format(n,time_of_flow))
				B_ICU[n] = self.dt * g.flow_ICU()


		# Increase the bounce level so as not to violate C^H / C^ICU
		# check whether there is overflow in H
		remaining_H_patients = sum([g.H[time_of_flow] * (1-self.dt * g.parameters["lambda_H_D"] - self.dt * g.parameters["lambda_H_R"])  for n,g in self.groups.items()])

		total_inflow_H_after_bounce = sum([self.dt * g.flow_H() - B_H[n]  for n,g in self.groups.items()])

		if( remaining_H_patients + total_inflow_H_after_bounce > self.beds * self.dt):
			# extra bounces done proportionally in each group
			# print("\nWARNING.group.py() Total entering H at t=%d = %.2f > remaining available capacity = %d. Bouncing more, proportionally." %(time_of_flow,total_inflow_H_after_bounce,self.beds * self.dt -remaining_H_patients) )

			extra_bounced = (1+tol) * (remaining_H_patients + total_inflow_H_after_bounce - self.beds * self.dt)
			for n,g in self.groups.items():
				B_H[n] += extra_bounced * (self.dt * g.flow_H()-B_H[n])/(total_inflow_H_after_bounce if total_inflow_H_after_bounce > 0 else 10e-10)

		# check whether there is overflow in ICU
		remaining_ICU_patients = sum([g.ICU[time_of_flow] * (1-self.dt * g.parameters["lambda_ICU_D"] - self.dt * g.parameters["lambda_ICU_R"])  for n,g in self.groups.items()])

		total_inflow_ICU_after_bounce = sum([self.dt * g.flow_ICU() - B_ICU[n]  for n,g in self.groups.items()])

		if( remaining_ICU_patients + total_inflow_ICU_after_bounce > self.icus * self.dt ):
			# extra bounces done proportionally in each group
			# print("\nWARNING.group.py() Total entering ICU at t=%d = %.2f > remaining available capacity = %d. Bouncing more, proportionally." %(time_of_flow,total_inflow_ICU_after_bounce,self.icus * self.dt -remaining_ICU_patients) )

			extra_bounced = (1+tol) * (remaining_ICU_patients + total_inflow_ICU_after_bounce - self.icus * self.dt)
			for n,g in self.groups.items():
				B_ICU[n] += extra_bounced * (self.dt * g.flow_ICU()-B_ICU[n])/(total_inflow_ICU_after_bounce if total_inflow_ICU_after_bounce > 0 else 10e-10)

		total_inflow_ICU_after_new_bounce = sum([self.dt * g.flow_ICU() - B_ICU[n]  for n,g in self.groups.items()])

		# print("After capping the Bouncing var total ICU should be: {}".format(remaining_ICU_patients + total_inflow_ICU_after_new_bounce))

		return B_H, B_ICU

	# Reset the overall simulation time
	def reset_time(self, new_time):
		"""Resets the current time of the simulation to an earlier time point"""


		assert(new_time <= self.t)


		self.use_gurobi_vars = False

		# reset the time in each group and check if we are using gurobi variables for the bouncing decisions.
		for n in self.groups:
			self.groups[n].reset_time(new_time)

			if(
			isinstance(self.groups[n].H[new_time], gb.Var)
			or
			isinstance(self.groups[n].ICU[new_time], gb.Var)
			or
			isinstance(self.groups[n].H[new_time], gb.LinExpr)
			or
			isinstance(self.groups[n].ICU[new_time], gb.LinExpr)):
				print("Reset Time: B_H and B_ICU are gurobi variables. Fixing the use_gurobi_vars flag to True.")
				self.use_gurobi_vars = True

		# reset internal calculations of econ values, deaths, rewards
		self.economic_values = self.economic_values[0:new_time+1]
		self.deaths = self.deaths[0:new_time+1]
		self.rewards = self.rewards[0:new_time+1]

		# reset controls to time new_time
		self.lockdown_controls = self.lockdown_controls[0:new_time]
		self.m_tests_controls = self.m_tests_controls[0:new_time]
		self.a_tests_controls = self.a_tests_controls[0:new_time]

		self.t = new_time

	# Simulates the dynamics given a vector of molecular tests, atomic tests and alphas
	def simulate(self, m_tests_vec, a_tests_vec, alphas_vec):
		for t in range(self.time_steps):
			self.take_time_step(m_tests_vec[t], a_tests_vec[t], alphas_vec[t])

	# Given a state and set of alphas, returns the economic value
	def get_economic_value(self, state, alphas):
		econ_activities = ["transport","leisure","other"]
		eta_activities = ["transport","leisure","other"]
		v_employment = 0
		v_schooling = 0
		self.v_g = {}
		self.econ_gradients = {"work":{},"school":{},"other":{},"leisure":{},"transport":{}}

		for age_group in state:
			v_g = sum([self.econ_params["employment_params"]["v"][age_group][activity] for activity in econ_activities])
			self.v_g[age_group] = v_g
			l_mean = np.mean([np.sum([alphas[ag][act] for act in eta_activities]) for ag in alphas])/3.0
			l_mean_upper_bound = np.sum([self.econ_params["upper_bounds"][act] for act in eta_activities])

			v_employment_g = 0
			v_employment_g += v_g*(
						self.econ_params["employment_params"]["nu"]*alphas[age_group]["work"]+
						self.econ_params["employment_params"]["eta"]*l_mean+
						self.econ_params["employment_params"]["gamma"]
					)*(state[age_group]["S"] + state[age_group]["E"] + state[age_group]["I"] + state[age_group]["R"])* self.dt

			# Add contribution of people fully recovered
			v_employment_g += v_g*(
						self.econ_params["employment_params"]["nu"]*1.0+
						self.econ_params["employment_params"]["eta"]*1.0+
						self.econ_params["employment_params"]["gamma"]
			)*state[age_group]["Rq"]* self.dt

			v_employment += v_employment_g
			# Add schooling contributions
			v_schooling += (

				self.experiment_params['delta_schooling']*self.econ_params['schooling_params'][age_group]*alphas[age_group]["school"]* 
				(state[age_group]["S"] + state[age_group]["E"] + state[age_group]["I"] + state[age_group]["R"])*self.dt+
				self.experiment_params['delta_schooling']*self.econ_params['schooling_params'][age_group]*1.0* 
				(state[age_group]["Rq"])*self.dt
			)

			# Calculate econ gradients
			self.econ_gradients["work"][age_group]=v_g*self.econ_params["employment_params"]["nu"]*(state[age_group]["S"] + state[age_group]["E"] + state[age_group]["I"] + state[age_group]["R"])
			self.econ_gradients["school"][age_group]=(self.experiment_params['delta_schooling']*self.econ_params['schooling_params'][age_group]* 
				(state[age_group]["S"] + state[age_group]["E"] + state[age_group]["I"] + state[age_group]["R"])*self.dt
			)
			for act in eta_activities:
				self.econ_gradients[act][age_group]=v_g*self.econ_params["employment_params"]["eta"]*(state[age_group]["S"] + state[age_group]["E"] + state[age_group]["I"] + state[age_group]["R"])/len(eta_activities)

		return v_employment + v_schooling


	def get_state(self, t):
		state = {}
		for name,group in self.groups.items():
			state[name] = {
				"S": group.S[t],
				"E": group.E[t],
				"I": group.I[t],
				"R": group.R[t],
				"N": group.N[t],
				"Ia": group.Ia[t],
				"Ips": group.Ips[t],
				"Ims": group.Ims[t],
				"Iss": group.Iss[t],
				"Rq": group.Rq[t],
				"H": group.H[t],
				"ICU": group.ICU[t],
				"D": group.D[t],
			}
		return state

	def get_delta_X_over_delta_t(self, t):
		assert(t>0)  # should never call this with t=0
		delta = {}
		for name,group in self.groups.items():
			delta[name] = {
				"S": (group.S[t] - group.S[t-1])/self.dt,
				"E": (group.E[t] - group.E[t-1])/self.dt,
				"I": (group.I[t] - group.I[t-1])/self.dt,
				"R": (group.R[t] - group.R[t-1])/self.dt,
				"N": (group.N[t] - group.N[t-1])/self.dt,
				"Ia": (group.Ia[t] - group.Ia[t-1])/self.dt,
				"Ips": (group.Ips[t] - group.Ips[t-1])/self.dt,
				"Ims": (group.Ims[t] - group.Ims[t-1])/self.dt,
				"Iss": (group.Iss[t] - group.Iss[t-1])/self.dt,
				"Rq": (group.Rq[t] - group.Rq[t-1])/self.dt,
				"H": (group.H[t] - group.H[t-1])/self.dt,
				"ICU": (group.ICU[t] - group.ICU[t-1])/self.dt,
				"D": (group.D[t] - group.D[t-1])/self.dt,
			}
		return delta


	def write_state(self, t, X):
		if t > self.t:
			for group_name in X.keys():
				self.groups[group_name].S += [0 for i in range(t - self.t)]
				self.groups[group_name].E+= [0 for i in range(t - self.t)]
				self.groups[group_name].I+= [0 for i in range(t - self.t)]
				self.groups[group_name].R+= [0 for i in range(t - self.t)]
				self.groups[group_name].N+= [0 for i in range(t - self.t)]
				self.groups[group_name].Ia+= [0 for i in range(t - self.t)]
				self.groups[group_name].Ips+= [0 for i in range(t - self.t)]
				self.groups[group_name].Ims+= [0 for i in range(t - self.t)]
				self.groups[group_name].Iss+= [0 for i in range(t - self.t)]
				self.groups[group_name].Rq+= [0 for i in range(t - self.t)]
				self.groups[group_name].H+= [0 for i in range(t - self.t)]
				self.groups[group_name].ICU+= [0 for i in range(t - self.t)]
				self.groups[group_name].D+= [0 for i in range(t - self.t)]
				self.groups[group_name].total_contacts += [0 for i in range(t - self.t)]
				self.groups[group_name].B_ICU += [0 for i in range(t - self.t)]
				self.groups[group_name].B_H += [0 for i in range(t - self.t)]
				
				

		for group_name in X.keys():
			self.groups[group_name].S[t] = X[group_name]['S']
			self.groups[group_name].E[t] = X[group_name]['E']
			self.groups[group_name].I[t] = X[group_name]['I']
			self.groups[group_name].R[t] = X[group_name]['R']
			self.groups[group_name].N[t] = X[group_name]['N']
			self.groups[group_name].Ia[t] = X[group_name]['Ia']
			self.groups[group_name].Ips[t] = X[group_name]['Ips']
			self.groups[group_name].Ims[t] = X[group_name]['Ims']
			self.groups[group_name].Iss[t] = X[group_name]['Iss']
			self.groups[group_name].Rq[t] = X[group_name]['Rq']
			self.groups[group_name].H[t] = X[group_name]['H']
			self.groups[group_name].ICU[t] = X[group_name]['ICU']
			self.groups[group_name].D[t] = X[group_name]['D']
		return 1

	def get_bounce(self, t):
		bounce_vars = {}
		for name,group in self.groups.items():
			bounce_vars[name] = {
				"B_H": group.B_H[t],
				"B_ICU": group.B_ICU[t],
			}
		return bounce_vars

	# Returns state but in OpenAIGym Format
	def get_normalized_state(self, t):
		norm_state = np.array([[
			group.S[t]/self.total_population,
			group.E[t]/self.total_population,
			group.I[t]/self.total_population,
			group.R[t]/self.total_population,
			(group.Ia[t] + group.Ips[t] + group.Ims[t])/self.total_population,
			group.Iss[t]/self.total_population,
			group.Rq[t]/self.total_population,
			group.H[t]/self.total_population,
			group.ICU[t]/self.total_population,
			group.D[t]/self.total_population
			] for name,group in self.groups.items()])
		return norm_state.flatten()


	# Returns state but in OpenAIGym Format
	def get_normalized_partial_state(self, t):
		norm_state = np.array([[
			(group.Ia[t] + group.Ips[t] + group.Ims[t])/self.total_population,
			group.Iss[t]/self.total_population,
			group.Rq[t]/self.total_population,
			group.H[t]/self.total_population,
			group.ICU[t]/self.total_population,
			group.D[t]/self.total_population
			] for name,group in self.groups.items()])
		return norm_state.flatten()

	def get_total_deaths(self, final_time_step=False):
		if final_time_step == False:
			final_time_step = self.time_steps
		total = 0
		for t in range(1,final_time_step+1):
			total += self.deaths[t]
		return total

	def get_total_economic_value(self, final_time_step=False):
		if final_time_step == False:
			final_time_step = self.time_steps
		total = 0
		for t in range(1,final_time_step+1):
			total += self.economic_values[t]
		return total

	def get_total_reward(self, final_time_step=False):
		if final_time_step == False:
			final_time_step = self.time_steps
		total = 0
		for t in range(1,final_time_step+1):
			total += self.rewards[t]
		return total

	def print_stats(self):
		print("Economic Value: "+str(self.get_total_economic_value()))
		print("Deaths "+str(self.get_total_deaths()))
		print("Total Reward "+str(self.get_total_reward()))

		for name, group in self.groups.items():
			deaths = 0
			no_days = len(group.D) - 1
			for day in range(no_days):
				deaths += group.D[day + 1]-group.D[day]
			print("Number of deaths for group {} is {}".format(name, deaths))

	def get_pandas_summary(self):
		d = {
			"S": [sum([self.groups[g].S[t] for g in self.groups ]) for t in range(self.t+1)],
			"E": [sum([self.groups[g].E[t] for g in self.groups ]) for t in range(self.t+1)],
			"I": [sum([self.groups[g].I[t] for g in self.groups ]) for t in range(self.t+1)],
			"R": [sum([self.groups[g].R[t] for g in self.groups ]) for t in range(self.t+1)],
			"N": [sum([self.groups[g].N[t] for g in self.groups ]) for t in range(self.t+1)],
			"Ia": [sum([self.groups[g].Ia[t] for g in self.groups ]) for t in range(self.t+1)],
			"Ips": [sum([self.groups[g].Ips[t] for g in self.groups ]) for t in range(self.t+1)],
			"Ims": [sum([self.groups[g].Ims[t] for g in self.groups ]) for t in range(self.t+1)],
			"Iss": [sum([self.groups[g].Iss[t] for g in self.groups ]) for t in range(self.t+1)],
			"Rq": [sum([self.groups[g].Rq[t] for g in self.groups ]) for t in range(self.t+1)],
			"H": [sum([self.groups[g].H[t] for g in self.groups ]) for t in range(self.t+1)],
			"ICU": [sum([self.groups[g].ICU[t] for g in self.groups ]) for t in range(self.t+1)],
			"D": [sum([self.groups[g].D[t] for g in self.groups ]) for t in range(self.t+1)],
		}
		return pd.DataFrame(d)

class SEIR_group:
	# Time step
	def __init__(self, group_parameters, group_initialization, dt, mixing_method, time_steps, parent):
		# Group name
		self.name = group_parameters['name']
		self.parameters = group_parameters['parameters']
		self.contacts = group_parameters['contacts']
		self.initial_conditions = group_initialization
		self.mixing_method = mixing_method
		self.time_steps = time_steps
		self.parent = parent
		self.initialize_vars(self.initial_conditions)

		# Time step
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
		self.IR = []

		# Bouncing variables
		self.B_H = []
		self.B_ICU = []


	def update_total_contacts(self, t, alphas):

		if (len(self.total_contacts) == t):
			summ_contacts = 0
			for n,g in self.all_groups.items():
				if(self.parent.use_gurobi_vars is True):
					# when using gurobi vars, set total population to initial N
					# print("WARNING. update_total_contacts(): Using initial population Ng(0) + Rgq(0) due to gurobi vars in dynModel.")
					pop_g = g.N[0] + g.Rq[0]
				else:
					pop_g = g.N[t] + g.Rq[t]
				new_contacts_dict = n_contacts(self, g, alphas, self.mixing_method)
				new_contacts = new_contacts_dict["total"]
				assert new_contacts >= 0, (f"New contacts is not nonnnegative: {new_contacts}.")
				summ_contacts += new_contacts*g.I[t]/(pop_g if pop_g!=0 else 10e-6)
				if self.parent.extra_data:
					self.parent.n_contacts[t][self.name][g.name] = new_contacts
					self.parent.n_infections[t][self.name][g.name] = new_contacts*g.I[t]/(pop_g if pop_g!=0 else 10e-6)*self.S[t]
					for act in all_activities:
						self.parent.n_infections_act[t][act][self.name][g.name] = new_contacts_dict[act]*g.I[t]/(pop_g if pop_g!=0 else 10e-6)*self.S[t]
			self.total_contacts.append(summ_contacts*self.S[t])
			self.IR.append(summ_contacts)

		else:
			print("t = ", t, "len of self.total_contacts = ", len(self.total_contacts))
			assert(False)


	# Attach other groups to make it easier to find variables of other groups
	def attach_other_groups(self,all_groups):
		self.all_groups = all_groups

	# Advances one time step, given the m_tests and a_tests variable
	def take_time_step(self, m_tests, a_tests, h_cap, icu_cap, B_H, B_ICU, B_ICU_perc = False):
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
		self.update_H(m_tests, a_tests, h_cap, icu_cap, B_H)
		self.update_ICU(m_tests, a_tests, h_cap, icu_cap, B_ICU, B_ICU_perc)
		self.update_D(m_tests, a_tests, h_cap, icu_cap, B_H, B_ICU, B_ICU_perc)


	# Reset the time to a past time
	def reset_time(self, new_time):
		assert(new_time <= self.parent.t)

		self.S = self.S[0:new_time+1]
		self.E = self.E[0:new_time+1]
		self.I = self.I[0:new_time+1]
		self.R = self.R[0:new_time+1]
		self.N = self.N[0:new_time+1]
		self.Ia = self.Ia[0:new_time+1]
		self.Ips = self.Ips[0:new_time+1]
		self.Ims = self.Ims[0:new_time+1]
		self.Iss = self.Iss[0:new_time+1]
		self.Rq = self.Rq[0:new_time+1]
		self.H = self.H[0:new_time+1]
		self.ICU = self.ICU[0:new_time+1]
		self.D = self.D[0:new_time+1]
		self.total_contacts = self.total_contacts[0:new_time]
		self.B_ICU = self.B_ICU[0:new_time]
		self.B_H = self.B_H[0:new_time]
		


	# Gives flow of how many people flowing to H
	# NOTE! time_of_flow here may correspond to ANOTHER group that is being updated, so
	# it may not be equal to self.t
	def flow_H(self):
		if self.parameters['p_H'] != 0.0:
			return  (self.parameters['mu']*self.parameters['p_H']*(self.I[self.parent.t]+self.Iss[self.parent.t]/(self.parameters['p_H']+self.parameters['p_ICU'])))
		else:
			return 0.0

	# Gives flow of how many people flowing to ICU
	# Same issue with time_of_flow as for H
	def flow_ICU(self):
		if self.parameters['p_ICU'] != 0.0:
			# print('flow_ICU(). In group {}, time from self is {}, time being updated is {}'.format(self.name,self.t,time_of_flow))
			return ( self.parameters['mu']*self.parameters['p_ICU']*(self.I[self.parent.t]+self.Iss[self.parent.t]/(self.parameters['p_H']+self.parameters['p_ICU'])))
		else:
			return 0.0

	# Updates N
	def update_N(self, m_tests, a_tests):
		delta_N = (
			- m_tests*self.I[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6)
			- a_tests*self.R[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6)
			- self.parameters['mu']*(self.parameters['p_H'] + self.parameters['p_ICU'])*self.I[self.parent.t]
		)
		self.N += [self.N[self.parent.t]+delta_N*self.dt]
		assert self.N[-1] >= 0, (f"Number of N is negative for t={self.parent.t+1}: {self.N[-1]}")

	# Updates S
	def update_S(self, m_tests, a_tests):
		delta_S = -self.parameters['beta'][self.parent.t]*self.total_contacts[self.parent.t]
		self.S += [self.S[self.parent.t]+delta_S*self.dt]
		assert self.S[-1] >= 0, (f"Number of S is negative for t={self.parent.t+1}: {self.S[-1]} \n Total contacts: {self.total_contacts}")

	# Updates Exposed
	def update_E(self, m_tests, a_tests):

		delta_E = self.parameters['beta'][self.parent.t]*self.total_contacts[self.parent.t] - self.parameters['sigma']*self.E[self.parent.t]
		assert self.E[self.parent.t] + delta_E * self.dt >=0, (f"Total Exposed is nonpositive: {self.E + delta_E * self.dt}. \n Delta E: {delta_E}. \n E at time t: {self.E[self.parent.t]}. \n Total contacts: {self.total_contacts[self.parent.t]}.") 
		
		self.E += [self.E[self.parent.t]+delta_E*self.dt]

		assert self.E[-1] >= 0, (f"Number of E is negative for t={self.parent.t+1}: {self.E[-1]}")


	# Updates infected
	def update_I(self, m_tests, a_tests):
		delta_I = self.parameters['sigma']*self.E[self.parent.t] - self.parameters['mu']*self.I[self.parent.t] - m_tests*self.I[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6)

		assert self.I[self.parent.t]+delta_I*self.dt >= 0, (f'Total infected is negative: {self.I[self.parent.t]+delta_I*self.dt}. \n Infected time t: {self.I[self.parent.t]}. \n Delta I: {delta_I}.\n Total N time t: {self.N[self.parent.t]}.\n Total E time t: {self.E[self.parent.t]}.\n M tests at time t: {m_tests}.\n Total m_tests: {m_tests * self.dt * self.I[self.parent.t]/self.N[self.parent.t]}.\n  Total new infected: {self.dt * self.parameters["sigma"]*self.E[self.parent.t]}.\n  Total recovered or going to H, ICU: {self.dt * self.parameters["mu"]*self.I[self.parent.t]}')

		self.I += [self.I[self.parent.t]+delta_I*self.dt]

		assert self.I[-1] >= 0, (f"Number of Infected is negative for t={self.parent.t+1}: {self.I[-1]}")


	# Updates recovered
	def update_R(self, m_tests, a_tests):
		delta_R = self.parameters['mu']*(1-self.parameters["p_H"]-self.parameters["p_ICU"])*self.I[self.parent.t] - a_tests*self.R[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6)
		self.R += [self.R[self.parent.t]+delta_R*self.dt]

		assert self.R[-1] >= 0, (f"Number of Recovered is negative for t={self.parent.t+1}: {self.R[-1]}")


	# Updates infected in quarantine
	def update_Ia(self, m_tests, a_tests):
		delta_Ia = self.parameters['p_Ia']*m_tests*self.I[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6) - self.parameters['mu']*self.Ia[self.parent.t]
		self.Ia += [self.Ia[self.parent.t]+delta_Ia*self.dt]

		assert self.Ia[-1] >= 0, (f"Number of Ia is negative for t={self.parent.t+1}: {self.Ia[-1]}")

	def update_Ips(self, m_tests, a_tests):
		delta_Ips = self.parameters['p_Ips']*m_tests*self.I[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6) - self.parameters['mu']*self.Ips[self.parent.t]
		self.Ips += [self.Ips[self.parent.t]+delta_Ips*self.dt]

		assert self.Ips[-1] >= 0, (f"Number of Ips is negative for t={self.parent.t+1}: {self.Ips[-1]}")

	def update_Ims(self, m_tests, a_tests):
		delta_Ims = self.parameters['p_Ims']*m_tests*self.I[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6) - self.parameters['mu']*self.Ims[self.parent.t]
		self.Ims += [self.Ims[self.parent.t]+delta_Ims*self.dt]
		
		assert self.Ims[-1] >= 0, (f"Number of Ims is negative for t={self.parent.t+1}: {self.Ims[-1]}")

	def update_Iss(self, m_tests, a_tests):
		delta_Iss = self.parameters['p_Iss']*m_tests*self.I[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6) - self.parameters['mu']*self.Iss[self.parent.t]
		self.Iss += [self.Iss[self.parent.t]+delta_Iss*self.dt]

		assert self.Iss[-1] >= 0, (f"Number of Iss is negative for t={self.parent.t+1}: {self.Iss[-1]}")


	# Update recovered in quarentine
	def update_Rq(self, m_tests, a_tests):
		delta_Rq = (
			self.parameters['mu']*(self.Ia[self.parent.t]+self.Ips[self.parent.t]+self.Ims[self.parent.t]) +
			self.parameters['lambda_H_R']*self.H[self.parent.t] +
			self.parameters['lambda_ICU_R']*self.ICU[self.parent.t] +
			a_tests*self.R[self.parent.t]/(self.N[self.parent.t] if self.N[self.parent.t]!=0 else 10e-6)
		)
		self.Rq += [self.Rq[self.parent.t]+delta_Rq*self.dt]

		assert self.Rq[-1] >= 0, (f"Number of Recovered Quarantined is negative for t={self.parent.t+1}: {self.Rq[-1]}")

	def update_H(self, m_tests, a_tests, h_cap, icu_cap, B_H):
		tol = 1e-9
		# For each group, calculate the entering amount
		entering_h = {}
		summ_entering_h = 0
		summ_staying_h = 0
		for n,g in self.all_groups.items():
			entering_h[n] = self.all_groups[n].flow_H()
			summ_entering_h += entering_h[n]
			summ_staying_h += (1-g.parameters['lambda_H_R']-g.parameters['lambda_H_D'])*g.H[self.parent.t]

		if B_H is False:
			B_H = entering_h[self.name]*((summ_entering_h-h_cap+summ_staying_h) if summ_entering_h-h_cap+summ_staying_h>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6)
			# ALTERNATIVE CLEANER EXPRESSION: (EASIER TO READ AND AVOIDS THE 1e-6 IN DENOMINATOR)
			# B_H = (0 if entering_h[self.name]==0 else (entering_h[self.name]/summ_entering_h*max(0,summ_entering_h-h_cap+summ_staying_h)))
		# Update bouncing variables
		self.B_H += [B_H]

		delta_H = (
			- (self.parameters["lambda_H_R"] + self.parameters["lambda_H_D"])*self.H[self.parent.t]
			+ (1-tol) * entering_h[self.name]

		)
		self.H += [self.H[self.parent.t]+delta_H*self.dt- B_H]

		assert self.H[-1] >= 0, (f"Number of Hospitalizations is negative for t={self.parent.t+1}: {self.H[-1]}")

	def update_ICU(self, m_tests, a_tests, h_cap, icu_cap, B_ICU, B_ICU_perc):
		tol = 1e-9

		# For each group, calculate the entering amount
		entering_icu = {}
		summ_entering_icu = 0
		summ_staying_icu = 0
		for n,g in self.all_groups.items():
			entering_icu[n] = self.all_groups[n].flow_ICU()
			summ_entering_icu += entering_icu[n]
			summ_staying_icu += (1-g.parameters['lambda_ICU_R']-g.parameters['lambda_ICU_D'])*g.ICU[self.parent.t]

		# print('update_ICU(): Total entering ICU calculated from group {} is:{}'.format(self.name,summ_entering_icu))

		if B_ICU_perc is not False:
			B_ICU = B_ICU_perc*((summ_entering_icu-icu_cap+summ_staying_icu) if summ_entering_icu-icu_cap+summ_staying_icu>0 else 0)
		elif B_ICU is False:
			#print("group.py(): FALSE branch for B_ICU")
			B_ICU = entering_icu[self.name]*((summ_entering_icu-icu_cap+summ_staying_icu) if summ_entering_icu-icu_cap+summ_staying_icu>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6)

		# Update bouncing variables
		self.B_ICU += [B_ICU]

		delta_ICU = (
			- (self.parameters["lambda_ICU_R"] + self.parameters["lambda_ICU_D"])*self.ICU[self.parent.t]
			+ (1-tol) * entering_icu[self.name]

		)
		self.ICU += [self.ICU[self.parent.t]+delta_ICU*self.dt - B_ICU]
		# print("update_ICU(): ICU occupancy at end of period t={} for group {} IS {}".format(self.parent.t, self.name,self.ICU[-1]))
		
		assert self.ICU[-1] >= 0, (f"Number of ICUs is negative for t={self.parent.t+1}: {self.ICU[-1]} \n B_ICU = {B_ICU} \n Delta ICU = {delta_ICU}")


	def update_D(self, m_tests, a_tests, h_cap, icu_cap, B_H, B_ICU, B_ICU_perc):
		# For each group, calculate the entering amount
		entering_h = {}
		summ_entering_h = 0
		summ_staying_h = 0
		for n,g in self.all_groups.items():
			entering_h[n] = self.all_groups[n].flow_H()
			summ_entering_h += entering_h[n]
			summ_staying_h += (1-g.parameters['lambda_H_R']-g.parameters['lambda_H_D'])*g.H[self.parent.t]

		entering_icu = {}
		summ_entering_icu = 0
		summ_staying_icu = 0
		for n,g in self.all_groups.items():
			entering_icu[n] = self.all_groups[n].flow_ICU()
			summ_entering_icu += entering_icu[n]
			summ_staying_icu += (1-g.parameters['lambda_ICU_R']-g.parameters['lambda_ICU_D'])*g.ICU[self.parent.t]

		if B_H is False:
			B_H = entering_h[self.name]*(summ_entering_h-h_cap+summ_staying_h if summ_entering_h-h_cap+summ_staying_h>0 else 0)/(summ_entering_h if summ_entering_h!=0 else 10e-6)
		if B_ICU_perc is not False:
			B_ICU = B_ICU_perc*((summ_entering_icu-icu_cap+summ_staying_icu) if summ_entering_icu-icu_cap+summ_staying_icu>0 else 0)
		elif B_ICU is False:
			B_ICU = entering_icu[self.name]*(summ_entering_icu-icu_cap+summ_staying_icu if summ_entering_icu-icu_cap+summ_staying_icu>0 else 0)/(summ_entering_icu if summ_entering_icu!=0 else 10e-6)


		delta_D = (
			self.parameters["lambda_H_D"]*self.H[self.parent.t]
			+ self.parameters["lambda_ICU_D"]*self.ICU[self.parent.t]

		)

		self.D += [self.D[self.parent.t]+delta_D*self.dt+ B_H
		+ B_ICU]

		assert self.D[-1] >= 0, (f"Number of deaths is negative for t={self.parent.t+1}: {self.D[-1]}")




