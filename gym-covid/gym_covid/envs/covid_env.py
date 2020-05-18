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


	def __init__(self, universe_params, simulation_params, actions_dict, initialization):
		super(CovidEnv, self).__init__()

		self.dt = simulation_params['dt']
		self.days = simulation_params['days']
		self.policy_freq = simulation_params['policy_freq']
		self.time_periods = int(round(self.days/self.dt))
		self.simulation_params = simulation_params
		self.actions_dict = actions_dict
		self.universe_params = universe_params

		# Construct model
		self.dynModel = DynamicalModel(universe_params, initialization, self.dt, self.time_periods)

		# Create action_space and observation space
		self.action_space = spaces.MultiDiscrete([len(actions_dict["age_group_"+str(i)]) for i in range(1,7)])
		self.observation_space = spaces.Box(low=0, high=1.0, shape=(61,), dtype=np.float32)


	def reset(self, initialization, tests):

		self.t = 0
		self.dynModel = DynamicalModel(self.universe_params, initialization, self.dt, self.time_periods)
		self.a_tests_vec = tests["a_tests_vec"]
		self.m_tests_vec = tests["m_tests_vec"]

		return np.append(self.dynModel.get_normalized_state(0),0)


	def step(self, action):

		alphas = {
			'age_group_%d'%i: self.actions_dict['age_group_%d'%i][action[i-1]] for i in range(1,7)
		}

		sum_rewards = 0
		for s in range(int(self.policy_freq/self.dt)):
			result = self.dynModel.take_time_step(self.m_tests_vec[self.t], self.a_tests_vec[self.t], alphas)
			sum_rewards += result['reward']
			self.t += 1
			done = (self.t == self.time_periods)
			if done:
				break

		observation = np.append(self.dynModel.get_normalized_state(self.t),self.t/(self.time_periods+0.0))		

		return observation, sum_rewards, done, {}

	def close(self):
		pass

	def same_lockdown_action(self,a):
		return [a for i in range(6)]







