import yaml
from group import SEIR_group, DynamicalModel

with open("parameters.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)

# Set up parameters of simulation
dt = 1
time_periods = 20
h_cap_vec = [parameters['global-parameters']['C_H'] for t in range(time_periods)]
icu_cap_vec = [parameters['global-parameters']['C_ICU'] for t in range(time_periods)]

m_tests_vec = [10 for t in range(time_periods)]
a_tests_vec = [10 for t in range(time_periods)]


# Simulate model
dynModel = DynamicalModel(parameters, dt)
dynModel.simulate(time_periods, m_tests_vec, a_tests_vec, h_cap_vec, icu_cap_vec)

# Calculate upper bound
# dynModel = DynamicalModel(parameters, dt)
# dynModel.upper_bound(time_periods, m_tests_vec, a_tests_vec, h_cap_vec, icu_cap_vec)
