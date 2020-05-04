import yaml
from inspect import getsourcefile
import os.path
import sys
import matplotlib
import matplotlib.pyplot as plt
import argparse

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
total_time = 80

time_periods = int(round(total_time/dt))



# Load number of beds, icus and tests
h_cap_vec = [parameters['global-parameters']['C_H'] for t in range(time_periods)]
icu_cap_vec = [parameters['global-parameters']['C_ICU'] for t in range(time_periods)]

m_tests_vec = [parameters['global-parameters']['M_tests'] for t in range(time_periods)]
a_tests_vec = [parameters['global-parameters']['A_tests'] for t in range(time_periods)]


# Simulate model
dynModel = DynamicalModel(parameters, dt)
dynModel.simulate(time_periods, m_tests_vec, a_tests_vec, h_cap_vec, icu_cap_vec)

# Draw plots
time_axis = [i*dt for i in range(time_periods+1)]


plt.figure(1)
plt.subplot(3,2,1)
plt.plot(time_axis, dynModel.groups['young'].S, label="Susceptible")
plt.plot(time_axis, dynModel.groups['young'].E, label="Exposed")
plt.plot(time_axis, dynModel.groups['young'].I, label="Infected")
plt.plot(time_axis, dynModel.groups['young'].R, label="Recovered")
plt.title('Young')

plt.subplot(3,2,2)
plt.plot(time_axis, dynModel.groups['old'].S, label="Susceptible")
plt.plot(time_axis, dynModel.groups['old'].E, label="Exposed")
plt.plot(time_axis, dynModel.groups['old'].I, label="Infected")
plt.plot(time_axis, dynModel.groups['old'].R, label="Recovered")
plt.legend(loc='upper right')
plt.title('Old')

plt.subplot(3,2,3)
plt.plot(time_axis, dynModel.groups['young'].Rq, label="Recovered Q")
plt.plot(time_axis, dynModel.groups['young'].Ia, label="Infected A-Q")
plt.plot(time_axis, dynModel.groups['young'].Ips, label="Infected PS-Q")
plt.plot(time_axis, dynModel.groups['young'].Ims, label="Infected MS-Q")
plt.plot(time_axis, dynModel.groups['young'].Iss, label="Infected SS-Q")
plt.xlabel('Time')

plt.subplot(3,2,4)
plt.plot(time_axis, dynModel.groups['old'].Rq, label="Recovered Q")
plt.plot(time_axis, dynModel.groups['old'].Ia, label="Infected A-Q")
plt.plot(time_axis, dynModel.groups['old'].Ips, label="Infected PS-Q")
plt.plot(time_axis, dynModel.groups['old'].Ims, label="Infected MS-Q")
plt.plot(time_axis, dynModel.groups['old'].Iss, label="Infected SS-Q")
plt.legend(loc='upper right')
plt.xlabel('Time')


plt.subplot(3,2,5)
plt.plot(time_axis, dynModel.groups['young'].H, label="Hospital Bed")
plt.plot(time_axis, dynModel.groups['young'].ICU, label="ICU")
plt.plot(time_axis, dynModel.groups['young'].D, label="Dead")
plt.xlabel('Time')


plt.subplot(3,2,6)
plt.plot(time_axis, dynModel.groups['old'].H, label="Hospital Bed")
plt.plot(time_axis, dynModel.groups['old'].ICU, label="ICU")
plt.plot(time_axis, dynModel.groups['old'].D, label="Dead")
plt.legend(loc='upper right')
plt.xlabel('Time')


figure = plt.gcf() # get current figure
figure.set_size_inches(12, 12)
plt.savefig(args.data.split(".")[0]+".png", dpi = 100)

