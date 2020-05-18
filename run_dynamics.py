import yaml
from inspect import getsourcefile
import os.path
import sys
import matplotlib
import matplotlib.pyplot as plt
import argparse
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
sys.path.insert(0, current_dir+"/heuristics")
from group import SEIR_group, DynamicalModel
from heuristics import *

# Global variables
dt = 0.1
days = 180
region = "Ile-de-France"

# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-lockdown", "--lockdown", help="Lockdown pattern")
parser.add_argument("-heuristic", "--heuristic", help="Whether to draw plots")
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
args = parser.parse_args()

print("read_all")

# Read group parameters
with open("./parameters/"+region+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("./initialization/initialization.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    initialization = yaml.load(file, Loader=yaml.FullLoader)

# Read lockdown
with open("./alphas/lockdown_%s.yaml"%args.lockdown) as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    alphas = yaml.load(file, Loader=yaml.FullLoader)



# Set up parameters of simulation
total_time = int(days)
time_periods = int(round(total_time/dt))


# Create model
dynModel = DynamicalModel(parameters, initialization, dt, time_periods)

# Construct vector of tests
max_m_tests = [float(args.m_tests) for t in range(time_periods)]
max_a_tests = [float(args.a_tests) for t in range(time_periods)]
if args.heuristic == "random":
	a_tests_vec, m_tests_vec = random_partition(dynModel, max_a_tests, max_m_tests)
elif args.heuristic == "homogeneous":
	a_tests_vec, m_tests_vec = homogeneous(dynModel, max_a_tests, max_m_tests)
elif "age_group" in args.heuristic:
	a_tests_vec, m_tests_vec = all_to_one(dynModel, args.heuristic, max_a_tests, max_m_tests)
elif args.heuristic == "no_tests":
	a_tests_vec, m_tests_vec = no_tests(dynModel)
elif args.heuristic == "forecasting_heuristic":
    tolerance = 1000000
    max_iterations = 2
    a_tests_vec, m_tests_vec = forecasting_heuristic(dynModel, max_a_tests, max_m_tests, h_cap_vec, icu_cap_vec, tolerance, max_iterations)


# Construct vector of alphas
alphas_vec = [alphas for t in range(time_periods)]


# Simulate model
dynModel.simulate(m_tests_vec, a_tests_vec, alphas_vec)
dynModel.print_stats()




# # string_to_print = str(dynModel.get_economic_value()) +"\t"+ str(dynModel.get_deaths())+"\t"+ str(dynModel.get_economic_value()-1000000*dynModel.get_deaths())
# # print(string_to_print)


# Draw plots
time_axis = [i*dt for i in range(time_periods+1)]


groups = dynModel.groups.keys()
plt.figure(1)
for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1)
	plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
	plt.title(group)
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1+len(groups))
	plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
	plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
	plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1+len(groups)*2)
	plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1+len(groups)*3)
	plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
	plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
	plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
	plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(6,len(groups),i+1+len(groups)*4)
	plt.plot(time_axis, dynModel.groups[group].H, label="Hospital Bed")
	plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
	plt.plot(time_axis, dynModel.groups[group].D, label="Dead")
	plt.legend(loc='upper right')

plt.subplot(6,2,11)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.axhline(y=parameters['global-parameters']['C_ICU'], color='g', linestyle='dashed', label= "ICU Capacity")
plt.legend(loc='upper right')

plt.subplot(6,2,12)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.legend(loc='upper right')



figure = plt.gcf() # get current figure
figure.set_size_inches(6*len(groups),18)
figure.suptitle('Region: %s, Lockdown-Pattern: %s, MTests/day: %s, Heuristic: %s'%(region,args.lockdown,args.m_tests,args.heuristic), fontsize=22)
plt.savefig("results_runs/"+region+"_lp_"+args.lockdown+"_m_tests_"+args.m_tests+"_heuristic_"+args.heuristic+".png", dpi = 100)

