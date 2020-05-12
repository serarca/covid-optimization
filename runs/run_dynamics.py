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
sys.path.insert(0, parent_dir+"/lower_bounds")
from group import SEIR_group, DynamicalModel
from heuristics import *


# Parse data
parser = argparse.ArgumentParser()
parser.add_argument("-region", "--region", help="Region")
parser.add_argument("-lockdown", "--lockdown", help="Lockdown pattern")
parser.add_argument("-plots", "--plots", help="Whether to draw plots")
parser.add_argument("-heuristic", "--heuristic", help="Whether to draw plots")
args = parser.parse_args()


with open("../parameters/"+args.region+"_lp_"+args.lockdown+"_params.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)

# Set up parameters of simulation
dt = 0.1
total_time = 180

time_periods = int(round(total_time/dt))

# Nonzero groups
groups = []
for group in parameters['seir-groups']:
	pop = sum([parameters['seir-groups'][group]['initial-conditions'][sg] for sg in ["S","E","I","R","Ia","Ips","Ims","Iss","Rq","H","ICU","D"]])
	if pop>0:
		groups.append(group)
groups.sort()
print(groups)

# Load number of beds, icus and tests
h_cap_vec = [parameters['global-parameters']['C_H'] for t in range(time_periods)]
icu_cap_vec = [parameters['global-parameters']['C_ICU'] for t in range(time_periods)]

max_m_tests = [parameters['global-parameters']['M_tests'] for t in range(time_periods)]
max_a_tests = [parameters['global-parameters']['A_tests'] for t in range(time_periods)]


# Create model
dynModel = DynamicalModel(parameters, dt, time_periods)

if args.heuristic == "random":
	m_tests_vec, a_tests_vec = random_partition(dynModel, groups, max_a_tests, max_m_tests)
elif args.heuristic == "homogeneous":
	m_tests_vec, a_tests_vec = homogeneous(dynModel, groups, max_a_tests, max_m_tests)
elif "age_group" in args.heuristic:
	if args.heuristic == "age_group_6" and args.lockdown>2:
		m_tests_vec, a_tests_vec = all_to_one(dynModel, args.heuristic+"_lp_2", max_a_tests, max_m_tests)
	else:
		m_tests_vec, a_tests_vec = all_to_one(dynModel, args.heuristic+"_lp_"+args.lockdown, max_a_tests, max_m_tests)
elif args.heuristic == "no_tests":
	m_tests_vec, a_tests_vec = no_tests(dynModel)


# Simulate model
dynModel.simulate(m_tests_vec, a_tests_vec, h_cap_vec, icu_cap_vec)
dynModel.print_stats()
deaths_per_group = [dynModel.groups[group].D[time_periods-1] for group in groups]

string_to_print = str(dynModel.get_economic_value())
for d in deaths_per_group:
	string_to_print += "\t"+str(d)
print(string_to_print)


# Draw plots
time_axis = [i*dt for i in range(time_periods+1)]



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
figure.suptitle('Region: %s, Lockdown-Pattern: %s'%(args.region,args.lockdown), fontsize=22)
plt.savefig(args.region+"_lp_"+args.lockdown+".png", dpi = 100)



# plt.figure(1)
# plt.subplot(5,2,1)
# plt.plot(time_axis, dynModel.groups['young'].S, label="Susceptible")
# plt.plot(time_axis, dynModel.groups['young'].E, label="Exposed")
# plt.plot(time_axis, dynModel.groups['young'].I, label="Infected")
# plt.plot(time_axis, dynModel.groups['young'].R, label="Recovered")
# plt.title('Young')

# plt.subplot(5,2,2)
# plt.plot(time_axis, dynModel.groups['old'].S, label="Susceptible")
# plt.plot(time_axis, dynModel.groups['old'].E, label="Exposed")
# plt.plot(time_axis, dynModel.groups['old'].I, label="Infected")
# plt.plot(time_axis, dynModel.groups['old'].R, label="Recovered")
# plt.legend(loc='upper right')
# plt.title('Old')

# plt.subplot(5,2,3)
# plt.plot(time_axis, dynModel.groups['young'].Rq, label="Recovered Q")
# plt.xlabel('Time')

# plt.subplot(5,2,4)
# plt.plot(time_axis, dynModel.groups['old'].Rq, label="Recovered Q")
# plt.legend(loc='upper right')
# plt.xlabel('Time')

# plt.subplot(5,2,5)
# plt.plot(time_axis, dynModel.groups['young'].Ia, label="Infected A-Q")
# plt.plot(time_axis, dynModel.groups['young'].Ips, label="Infected PS-Q")
# plt.plot(time_axis, dynModel.groups['young'].Ims, label="Infected MS-Q")
# plt.plot(time_axis, dynModel.groups['young'].Iss, label="Infected SS-Q")
# plt.xlabel('Time')

# plt.subplot(5,2,6)
# plt.plot(time_axis, dynModel.groups['old'].Ia, label="Infected A-Q")
# plt.plot(time_axis, dynModel.groups['old'].Ips, label="Infected PS-Q")
# plt.plot(time_axis, dynModel.groups['old'].Ims, label="Infected MS-Q")
# plt.plot(time_axis, dynModel.groups['old'].Iss, label="Infected SS-Q")
# plt.legend(loc='upper right')
# plt.xlabel('Time')


# plt.subplot(5,2,7)
# plt.plot(time_axis, dynModel.groups['young'].H, label="Hospital Bed")
# plt.plot(time_axis, dynModel.groups['young'].ICU, label="ICU")
# plt.plot(time_axis, dynModel.groups['young'].D, label="Dead")
# plt.xlabel('Time')


# plt.subplot(5,2,8)
# plt.plot(time_axis, dynModel.groups['old'].H, label="Hospital Bed")
# plt.plot(time_axis, dynModel.groups['old'].ICU, label="ICU")
# plt.plot(time_axis, dynModel.groups['old'].D, label="Dead")
# plt.legend(loc='upper right')
# plt.xlabel('Time')

# plt.subplot(5,1,5)
# plt.plot(time_axis, [dynModel.groups['old'].H[i] + dynModel.groups['young'].H[i] for i in range(len(time_axis))], label="Total Hospital Beds")
# plt.plot(time_axis, [dynModel.groups['old'].ICU[i] + dynModel.groups['young'].ICU[i] for i in range(len(time_axis))], label="Total ICUs")
# plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
# plt.axhline(y=parameters['global-parameters']['C_ICU'], color='g', linestyle='dashed', label= "ICU Capacity")
# plt.legend(loc='upper right')
# plt.xlabel('Time')

# figure = plt.gcf() # get current figure
# figure.set_size_inches(12, 12)
# plt.savefig(args.data.split(".")[0]+".png", dpi = 100)


