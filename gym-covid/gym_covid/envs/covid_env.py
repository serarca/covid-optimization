import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os.path
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))
import sys
sys.path.insert(0, "../../../../../covid-optimization")
sys.path.insert(0, "../../../../../covid-optimization/heuristics")
sys.path.insert(0, "../../covid-optimization")
sys.path.insert(0, "../../covid-optimization/heuristics")


from group import SEIR_group, DynamicalModel
from heuristics import *

import numpy as np
import yaml
import math


class CovidEnvMultiDiscrete(gym.Env):


	def __init__(self, universe_params, simulation_params, actions_dict, initialization):
		super(CovidEnvMultiDiscrete, self).__init__()

		self.dt = simulation_params['dt']
		self.days = simulation_params['days']
		self.policy_freq = simulation_params['policy_freq']
		self.time_periods = int(round(self.days/self.dt))
		self.simulation_params = simulation_params
		self.actions_dict = actions_dict
		self.universe_params = universe_params
		self.initialization = initialization

		# Construct model
		self.dynModel = DynamicalModel(universe_params, initialization, self.dt, self.time_periods)

		# Create action_space and observation space
		self.action_space = spaces.MultiDiscrete([len(actions_dict["age_group_"+str(i)]) for i in range(1,7)])
		self.observation_space = spaces.Box(low=0, high=1.0, shape=(61,), dtype=np.float32)

	def testing(self, tests):
		self.tests = tests

	def reset(self):

		self.t = 0
		self.dynModel = DynamicalModel(self.universe_params, self.initialization, self.dt, self.time_periods)
		self.a_tests_vec = self.tests["a_tests_vec"]
		self.m_tests_vec = self.tests["m_tests_vec"]

		return np.append(self.dynModel.get_normalized_state(0),0)


	def step(self, action):

		alphas = {
			'age_group_%d'%i: self.actions_dict['age_group_%d'%i][action[i-1]] for i in range(1,7)
		}

		sum_rewards = 0
		for s in range(int(self.policy_freq/self.dt)):
			result = self.dynModel.take_time_step(self.m_tests_vec[int(int(self.t)/int(self.policy_freq/self.dt))], self.a_tests_vec[int(int(self.t)/int(self.policy_freq/self.dt))], alphas)
			sum_rewards += result['reward']
			self.t += 1
			done = (self.t == self.time_periods)
			if done:
				break

		observation = np.append(self.dynModel.get_normalized_state(self.t),self.t/(self.time_periods+0.0))		

		return observation, sum_rewards, done, {}

	def close(self):
		pass


class CovidEnvDiscrete(gym.Env):


	def __init__(self, universe_params, simulation_params, actions_dict, initialization):
		super(CovidEnvDiscrete, self).__init__()

		self.dt = simulation_params['dt']
		self.days = simulation_params['days']
		self.policy_freq = simulation_params['policy_freq']
		self.time_periods = int(round(self.days/self.dt))
		self.simulation_params = simulation_params
		self.actions_dict = actions_dict
		self.universe_params = universe_params
		self.initialization = initialization

		# Construct model
		self.dynModel = DynamicalModel(universe_params, initialization, self.dt, self.time_periods)

		# Create action_space and observation space
		n_actions = 1
		for i in range(1,7):
			n_actions = n_actions*len(actions_dict["age_group_"+str(i)])
		self.action_space = spaces.Discrete(n_actions)
		self.observation_space = spaces.Box(low=0, high=1.0, shape=(61,), dtype=np.float32)

		self.mapping = []
		self.inv_mapping = {}
		counter = 0
		for i1 in range(len(self.actions_dict['age_group_1'])):
			for i2 in range(len(self.actions_dict['age_group_2'])):
				for i3 in range(len(self.actions_dict['age_group_3'])):
					for i4 in range(len(self.actions_dict['age_group_4'])):
						for i5 in range(len(self.actions_dict['age_group_5'])):
							for i6 in range(len(self.actions_dict['age_group_6'])):
								self.mapping.append((i1,i2,i3,i4,i5,i6))
								self.inv_mapping[(i1,i2,i3,i4,i5,i6)] = counter
								counter += 1

	def testing(self, tests):
		self.tests = tests

	def reset(self):

		self.t = 0
		self.dynModel = DynamicalModel(self.universe_params, self.initialization, self.dt, self.time_periods)
		self.a_tests_vec = self.tests["a_tests_vec"]
		self.m_tests_vec = self.tests["m_tests_vec"]

		return np.append(self.dynModel.get_normalized_state(0),0)


	def step(self, action):

		multid_action = self.action_to_multidiscrete(action)
		alphas = {
			'age_group_%d'%i: self.actions_dict['age_group_%d'%i][multid_action[i-1]] for i in range(1,7)
		}

		sum_rewards = 0
		for s in range(int(self.policy_freq/self.dt)):
			result = self.dynModel.take_time_step(self.m_tests_vec[int(int(self.t)/int(self.policy_freq/self.dt))], self.a_tests_vec[int(int(self.t)/int(self.policy_freq/self.dt))], alphas)
			sum_rewards += result['reward']
			self.t += 1
			done = (self.t == self.time_periods)
			if done:
				break

		observation = np.append(self.dynModel.get_normalized_state(self.t),self.t/(self.time_periods+0.0))		

		return observation, sum_rewards, done, {}

	def close(self):
		pass

	def action_to_multidiscrete(self, action):
		return self.mapping[action]

	def multidiscrete_to_action(self, multid):
		return self.inv_mapping[multid]


class CovidEnvContinuous(gym.Env):

	def __init__(self, universe_params, simulation_params, actions_dict, initialization):
		super(CovidEnvContinuous, self).__init__()

		self.dt = simulation_params['dt']
		self.days = simulation_params['days']
		self.policy_freq = simulation_params['policy_freq']
		self.time_periods = int(round(self.days/self.dt))
		self.simulation_params = simulation_params
		self.actions_dict = actions_dict
		self.universe_params = universe_params
		self.initialization = initialization

		# Construct model
		self.dynModel = DynamicalModel(universe_params, initialization, self.dt, self.time_periods)

		# Create action_space and observation space
		print([len(actions_dict['age_group_%d'%(i+1)])-1 for i in range(6)])
		self.action_space = spaces.Box(low=np.array([np.float32(0) for i in range(6)]), high=np.array([np.float32(len(actions_dict['age_group_%d'%(i+1)])-1) for i in range(6)]), shape=(6,), dtype=np.float32)
		self.observation_space = spaces.Box(low=0, high=1.0, shape=(61,), dtype=np.float32)


	def testing(self, tests):
		self.tests = tests

	def reset(self):

		self.t = 0
		self.dynModel = DynamicalModel(self.universe_params, self.initialization, self.dt, self.time_periods)
		self.a_tests_vec = self.tests["a_tests_vec"]
		self.m_tests_vec = self.tests["m_tests_vec"]

		return np.append(self.dynModel.get_normalized_state(0),0)


	def step(self, action):

		alphas = self.action_to_alphas(action)
		sum_rewards = 0
		for s in range(int(self.policy_freq/self.dt)):
			result = self.dynModel.take_time_step(self.m_tests_vec[int(int(self.t)/int(self.policy_freq/self.dt))], self.a_tests_vec[int(int(self.t)/int(self.policy_freq/self.dt))], alphas)
			sum_rewards += result['reward']
			self.t += 1
			done = (self.t == self.time_periods)
			if done:
				break

		observation = np.append(self.dynModel.get_normalized_state(self.t),self.t/(self.time_periods+0.0))		

		return observation, sum_rewards, done, {}

	def close(self):
		pass

	def action_to_alphas(self, action):
		alphas = {}
		for i in range(6):
			a = action[i]
			lower = math.floor(a)
			upper = math.ceil(a)
			coeff = a-lower
			alphas['age_group_%d'%(i+1)] = {}
			for activity in self.actions_dict['age_group_%d'%(i+1)][0]:
				alphas['age_group_%d'%(i+1)][activity] = self.actions_dict['age_group_%d'%(i+1)][int(lower)][activity] * (1-coeff) + self.actions_dict['age_group_%d'%(i+1)][int(upper)][activity] * coeff
		return alphas







