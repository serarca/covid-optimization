from gym.envs.registration import register

register(
    id='covid-discrete-v0',
    entry_point='gym_covid.envs:CovidEnvDiscrete',
)

register(
    id='covid-multidiscrete-v0',
    entry_point='gym_covid.envs:CovidEnvMultiDiscrete',
)

register(
    id='covid-continuous-v0',
    entry_point='gym_covid.envs:CovidEnvContinuous',
)