import gym
from stable_baselines.common.env_checker import check_env

from gym_covid.envs.covid_env import CovidEnv

from stable_baselines import A2C
from stable_baselines.common.cmd_util import make_vec_env


steps = 100000


# Instantiate the env
env = CovidEnv()

# wrap it
env = make_vec_env(lambda: env, n_envs=1)
model = A2C('MlpPolicy', env, verbose=1)

# Epochs
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

model.save("a2c_model")