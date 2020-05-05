import yaml
from inspect import getsourcefile
import os.path
import sys
import matplotlib
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
import numpy as np

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
from group import SEIR_group, DynamicalModel

# Parse data
parser = argparse.ArgumentParser()
parser.add_argument("-data", "--data", help="Source file for data")
args = parser.parse_args()


with open(args.data) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)

# Set up parameters of simulation
dt = 0.1
total_time = 100

time_periods = int(round(total_time/dt))


# Load number of beds, icus
h_cap_vec = [parameters['global-parameters']['C_H'] for t in range(time_periods)]
icu_cap_vec = [parameters['global-parameters']['C_ICU'] for t in range(time_periods)]

# Create vector of maximum number of tests at each time step
max_m_tests_vec = [parameters['global-parameters']['M_tests'] for t in range(time_periods)]
max_a_tests_vec = [parameters['global-parameters']['A_tests'] for t in range(time_periods)]


def sample_rho_trejectory_uniform(time_periods, parameters, dt, max_m_tests_vec, max_a_tests_vec, h_cap_vec, icu_cap_vec):
	dynModel = DynamicalModel(parameters, dt)
	a_sample = defaultdict(list)
	m_sample = defaultdict(list)
	# Sample dictionary of A tests for all groups at all times uniformly from the simplex boundary
	for t in range(time_periods):
		sample_sum = 0
		for n in parameters['seir-groups']:
			sample = np.random.uniform()
			a_sample[n].append(sample)
			sample_sum += sample
		for n in parameters['seir-groups']:
			a_sample[n][t] = a_sample[n][t]/sample_sum*max_a_tests_vec[t]
	# Sample dictionary of M tests for all groups at all times uniformly from the simplex boundary
	for t in range(time_periods):
		sample_sum = 0
		for n in parameters['seir-groups']:
			sample = np.random.uniform()
			m_sample[n].append(sample)
			sample_sum += sample
		for n in parameters['seir-groups']:
			m_sample[n][t] = m_sample[n][t]/sample_sum*max_m_tests_vec[t]

	# Simulate with the given samples
	dynModel.simulate(time_periods, m_sample, a_sample, h_cap_vec, icu_cap_vec)

	result = {
		n:dynModel.groups[n].rho for n in dynModel.groups
	}
	return result


# Choose a group to plot
group = 'young'
plt.figure(1)
time_axis = [i*dt for i in range(time_periods+1)]
for i in range(10):
	y = sample_rho_trejectory_uniform(time_periods, parameters, dt, max_m_tests_vec, max_a_tests_vec, h_cap_vec, icu_cap_vec)[group]
	plt.plot(time_axis, y)


# Calculate bounds on rho
dynModel = DynamicalModel(parameters, dt)
rho = dynModel.groups[group]
rho_lb_vector, rho_ub_vector = dynModel.get_rho_bounds(time_periods)
plt.plot(time_axis, rho_lb_vector[group], linestyle='dashed', label="upper bound")
plt.plot(time_axis, rho_ub_vector[group], linestyle='dashed', label="lower bound")


plt.title('Rho samples')

figure = plt.gcf() # get current figure
figure.set_size_inches(6, 6)
plt.savefig('rho_'+args.data.split(".")[0]+".png", dpi = 100)




	