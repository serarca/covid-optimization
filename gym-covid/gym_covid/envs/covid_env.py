import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os.path
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))

import sys
sys.path.insert(0, "/Users/sergioacamelogomez/Dropbox/covid-optimization")
sys.path.insert(0, "/Users/sergioacamelogomez/Dropbox/covid-optimization/heuristics")

from group import SEIR_group, DynamicalModel
from heuristics import *

import numpy as np
import yaml


class CovidEnv(gym.Env):


	def __init__(self):
		super(CovidEnv, self).__init__()

		self.n_actions_space = 5
		self.action_space = spaces.MultiDiscrete([self.n_actions_space for i in range(6)])



		#self.action_space = spaces.Box(low=0, high=1.0, shape=(6,), dtype=np.float32)
		self.observation_space = spaces.Box(low=0, high=1.0, shape=(61,), dtype=np.float32)

		days = 182.0
		self.dt = 0.1

		self.policy_freq = 7

		# Read group parameters
		with open("/Users/sergioacamelogomez/Dropbox/covid-optimization/parameters/Ile-de-France.yaml") as file:
		    # The FullLoader parameter handles the conversion from YAML
		    # scalar values to Python the dictionary format
		    self.parameters = yaml.load(file, Loader=yaml.FullLoader)

		# Read initialization
		with open("/Users/sergioacamelogomez/Dropbox/covid-optimization/initialization/initialization.yaml") as file:
		    # The FullLoader parameter handles the conversion from YAML
		    # scalar values to Python the dictionary format
		    self.initialization = yaml.load(file, Loader=yaml.FullLoader)

		# Set up parameters of simulation
		total_time = int(days)
		self.time_periods = int(round(total_time/self.dt))

		self.dynModel = DynamicalModel(self.parameters, self.initialization, self.dt, self.time_periods)

		# Construct vector of tests
		max_m_tests = [float(0) for t in range(self.time_periods)]
		max_a_tests = [float(0) for t in range(self.time_periods)]
		a_tests_vec, m_tests_vec = homogeneous(self.dynModel, max_a_tests, max_m_tests)


		self.a_tests_vec = a_tests_vec
		self.m_tests_vec = m_tests_vec


	def reset(self):

		self.t = 0
		self.dynModel = DynamicalModel(self.parameters, self.initialization, self.dt, self.time_periods)

		return np.append(self.dynModel.get_normalized_state(0),0)


	def step(self, action):

		alphas = {
			'age_group_%d'%(i+1):{
				"home":1.0,
				"work":float(action[i])/(self.n_actions_space-1),
				"school":float(action[i])/(self.n_actions_space-1),
				"transport":float(action[i])/(self.n_actions_space-1),
				"leisure":float(action[i])/(self.n_actions_space-1),
				"other":float(action[i])/(self.n_actions_space-1),
			} for i in range(6)
		}

		# alphas = {
		# 	'age_group_%d'%(i+1):{
		# 		"home":1.0,
		# 		"work":float(action['age_group_%d'%(i+1)])/(self.n_actions_space-1),
		# 		"school":float(action['age_group_%d'%(i+1)])/(self.n_actions_space-1),
		# 		"transport":float(action['age_group_%d'%(i+1)])/(self.n_actions_space-1),
		# 		"leisure":float(action['age_group_%d'%(i+1)])/(self.n_actions_space-1),
		# 		"other":float(action['age_group_%d'%(i+1)])/(self.n_actions_space-1),
		# 	} for i in range(6)
		# }

		# alphas = {
		# 	'age_group_%d'%(i+1):{
		# 		"home":1.0,
		# 		"work":action[i],
		# 		"school":action[i],
		# 		"transport":action[i],
		# 		"leisure":action[i],
		# 		"other":action[i],
		# 	} for i in range(6)
		# }

		sum_rewards = 0
		for s in range(int(self.policy_freq/self.dt)):
			result = self.dynModel.take_time_step(self.m_tests_vec[self.t], self.a_tests_vec[self.t], alphas)
			sum_rewards += result['reward']
			self.t += 1
			done = (self.t == self.time_periods)
			if done:
				break

		observation = np.append(self.dynModel.get_normalized_state(self.t),self.t/self.time_periods)		

		return observation, sum_rewards, done, {}

	def close(self):
		pass

	def same_lockdown_action(self,a):
		return [a for i in range(6)]







