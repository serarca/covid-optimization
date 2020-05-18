from gym.envs.registration import register

register(
    id='covid-v0',
    entry_point='gym_covid.envs:CovidEnv',
)