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
total_time = 100

time_periods = int(round(total_time/dt))



# Load number of beds, icus and tests
h_cap_vec = [parameters['global-parameters']['C_H'] for t in range(time_periods)]
icu_cap_vec = [parameters['global-parameters']['C_ICU'] for t in range(time_periods)]

m_tests_vec = {n:[parameters['global-parameters']['M_tests'] for t in range(time_periods)] for n in parameters['seir-groups']}
a_tests_vec = {n:[parameters['global-parameters']['A_tests'] for t in range(time_periods)] for n in parameters['seir-groups']}



# Simulate model
dynModel = DynamicalModel(parameters, dt)
dynModel.simulate(time_periods, m_tests_vec, a_tests_vec, h_cap_vec, icu_cap_vec)


# Draw plots
time_axis = [i*dt for i in range(time_periods+1)]


plt.figure(1)
plt.subplot(5,2,1)
plt.plot(time_axis, dynModel.groups['young'].S, label="Susceptible")
plt.plot(time_axis, dynModel.groups['young'].E, label="Exposed")
plt.plot(time_axis, dynModel.groups['young'].I, label="Infected")
plt.plot(time_axis, dynModel.groups['young'].R, label="Recovered")
plt.title('Young')

plt.subplot(5,2,2)
plt.plot(time_axis, dynModel.groups['old'].S, label="Susceptible")
plt.plot(time_axis, dynModel.groups['old'].E, label="Exposed")
plt.plot(time_axis, dynModel.groups['old'].I, label="Infected")
plt.plot(time_axis, dynModel.groups['old'].R, label="Recovered")
plt.legend(loc='upper right')
plt.title('Old')

plt.subplot(5,2,3)
plt.plot(time_axis, dynModel.groups['young'].Rq, label="Recovered Q")
plt.xlabel('Time')

plt.subplot(5,2,4)
plt.plot(time_axis, dynModel.groups['old'].Rq, label="Recovered Q")
plt.legend(loc='upper right')
plt.xlabel('Time')

plt.subplot(5,2,5)
plt.plot(time_axis, dynModel.groups['young'].Ia, label="Infected A-Q")
plt.plot(time_axis, dynModel.groups['young'].Ips, label="Infected PS-Q")
plt.plot(time_axis, dynModel.groups['young'].Ims, label="Infected MS-Q")
plt.plot(time_axis, dynModel.groups['young'].Iss, label="Infected SS-Q")
plt.xlabel('Time')

plt.subplot(5,2,6)
plt.plot(time_axis, dynModel.groups['old'].Ia, label="Infected A-Q")
plt.plot(time_axis, dynModel.groups['old'].Ips, label="Infected PS-Q")
plt.plot(time_axis, dynModel.groups['old'].Ims, label="Infected MS-Q")
plt.plot(time_axis, dynModel.groups['old'].Iss, label="Infected SS-Q")
plt.legend(loc='upper right')
plt.xlabel('Time')


plt.subplot(5,2,7)
plt.plot(time_axis, dynModel.groups['young'].H, label="Hospital Bed")
plt.plot(time_axis, dynModel.groups['young'].ICU, label="ICU")
plt.plot(time_axis, dynModel.groups['young'].D, label="Dead")
plt.xlabel('Time')


plt.subplot(5,2,8)
plt.plot(time_axis, dynModel.groups['old'].H, label="Hospital Bed")
plt.plot(time_axis, dynModel.groups['old'].ICU, label="ICU")
plt.plot(time_axis, dynModel.groups['old'].D, label="Dead")
plt.legend(loc='upper right')
plt.xlabel('Time')

plt.subplot(5,1,5)
plt.plot(time_axis, [dynModel.groups['old'].H[i] + dynModel.groups['young'].H[i] for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [dynModel.groups['old'].ICU[i] + dynModel.groups['young'].ICU[i] for i in range(len(time_axis))], label="Total ICUs")
plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.axhline(y=parameters['global-parameters']['C_ICU'], color='g', linestyle='dashed', label= "ICU Capacity")
plt.legend(loc='upper right')
plt.xlabel('Time')

figure = plt.gcf() # get current figure
figure.set_size_inches(12, 12)
plt.savefig(args.data.split(".")[0]+".png", dpi = 100)


