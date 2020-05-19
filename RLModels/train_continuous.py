import gym
from stable_baselines.common.env_checker import check_env

from gym_covid.envs.covid_env import CovidEnvContinuous

from stable_baselines import SAC
from stable_baselines.common.cmd_util import make_vec_env
import math
import argparse
import yaml
import sys
sys.path.insert(0, "../../covid-optimization")
sys.path.insert(0, "../../covid-optimization/heuristics")

from group import SEIR_group, DynamicalModel
from heuristics import *



# Global variables
simulation_params = {
	'dt':1.0,
	'days': 182.0,
	'policy_freq': 7.0,
}
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))
simulation_params['n_policies'] = int(math.ceil(simulation_params["days"]/simulation_params["policy_freq"]))



region = "Ile-de-France"

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-heuristic", "--heuristic", help="Whether to draw plots")
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
parser.add_argument("-steps", "--steps", help="Steps for learning algorithm")
args = parser.parse_args()

steps = int(args.steps)



# Read group parameters
with open("../parameters/"+region+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("../initialization/initialization.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    initialization = yaml.load(file, Loader=yaml.FullLoader)

# Read lockdown
with open("../alphas_action_space/default.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    actions_dict = yaml.load(file, Loader=yaml.FullLoader)


# Instantiate the env
env = CovidEnvContinuous(universe_params, simulation_params, actions_dict, initialization)

# Construct vector of tests with a heuristic
max_m_tests = [float(args.m_tests) for t in range(simulation_params['n_policies'])]
max_a_tests = [float(args.a_tests) for t in range(simulation_params['n_policies'])]
if args.heuristic == "random":
	a_tests_vec, m_tests_vec = random_partition(env.dynModel, max_a_tests, max_m_tests)
elif args.heuristic == "homogeneous":
	a_tests_vec, m_tests_vec = homogeneous(env.dynModel, max_a_tests, max_m_tests)
elif "age_group" in args.heuristic:
	a_tests_vec, m_tests_vec = all_to_one(env.dynModel, args.heuristic, max_a_tests, max_m_tests)
elif args.heuristic == "no_tests":
	a_tests_vec, m_tests_vec = no_tests(env.dynModel)
elif args.heuristic == "forecasting_heuristic":
    tolerance = 1000000
    max_iterations = 2
    a_tests_vec, m_tests_vec = forecasting_heuristic(env.dynModel, max_a_tests, max_m_tests, h_cap_vec, icu_cap_vec, tolerance, max_iterations)

tests = {
	'a_tests_vec':a_tests_vec,
	'm_tests_vec':m_tests_vec,
}
env.testing(tests)


# wrap it
env = make_vec_env(lambda: env, n_envs=1)
model = A2C('MlpPolicy', env, verbose=1)

# Learn
model = model.learn(steps)

# Test
obs = env.reset()
a = 1
rewards = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    rewards += reward
    print(action)
    if done:
        break
print("Rewards: %f"%rewards)

model.save("a2c_continuous")