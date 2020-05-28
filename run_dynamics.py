import yaml
from inspect import getsourcefile
import os.path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
sys.path.insert(0, current_dir+"/heuristics")


from group import SEIR_group, DynamicalModel
# from heuristics import *
from forecasting_heuristic import *
import math
import pprint


# Global variables
simulation_params = {
	'dt':1.0,
	'days': 182.0,
	'region': "Ile-de-France",
}
age_groups = ['age_group_0_9', 'age_group_10_19', 'age_group_20_29', 'age_group_30_39', 'age_group_40_49',
	'age_group_50_59', 'age_group_60_69', 'age_group_70_79', 'age_group_80_plus']


# Define time variables
simulation_params['time_periods'] = int(math.ceil(simulation_params["days"]/simulation_params["dt"]))



# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument("-heuristic", "--heuristic", help="Chose heuristic")
parser.add_argument("-a_tests", "--a_tests", help="Number of A tests")
parser.add_argument("-m_tests", "--m_tests", help="Number of M tests")
parser.add_argument("-policy", "--policy", help="Parameters of policy")
parser.add_argument("-perc_infected", "--perc_infected", help="Percentage of population infected")
parser.add_argument("-mixing", "--mixing", help="Type of mixing (can be maxmin, max, min, or mult)")
parser.add_argument("-mixing_param", "--mixing_param", help="If the type of mixing is maxmin, then the mixing_param is the appropiate alpha")
args = parser.parse_args()


# Read group parameters
with open("./parameters/"+simulation_params["region"]+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("./initialization/initialization.yaml") as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	initialization = yaml.load(file, Loader=yaml.FullLoader)

# Read lockdown
with open("./alphas_action_space/default.yaml") as file:
	# The FullLoader parameter handles the conversion from YAML
	# scalar values to Python the dictionary format
	actions_dict = yaml.load(file, Loader=yaml.FullLoader)


# Move population to infected
for group in initialization:
	change = initialization[group]["S"]*float(args.perc_infected)/100
	initialization[group]["S"] = initialization[group]["S"] - change
	initialization[group]["I"] = initialization[group]["I"] + change

# Define policy
if args.policy in ["static","dynamic"]:
	# Read policy
	with open('./benchmarks/'+args.policy+"_infected_"+args.perc_infected+'.yaml') as file:
	    # The FullLoader parameter handles the conversion from YAML
	    # scalar values to Python the dictionary format
	    policy_file = yaml.load(file, Loader=yaml.FullLoader)
	alphas_vec = policy_file['alphas_vec']

else:
	static_alpha = {
		age_groups[i]:actions_dict[age_groups[i]][int(args.policy[i])] for i in range(len(age_groups))
	}
	alphas_vec = [static_alpha for t in range(simulation_params['time_periods'])]

# Define mixing parameter
mixing_method = {
	"name":args.mixing,
	"param":float(args.mixing_param) if args.mixing_param else 0.0,
}



# Create environment
dynModel = DynamicalModel(universe_params, initialization, simulation_params['dt'], simulation_params['time_periods'], mixing_method, extra_data = True)


# Construct vector of tests with a heuristic
max_m_tests = [float(args.m_tests) for t in range(simulation_params['time_periods'])]
max_a_tests = [float(args.a_tests) for t in range(simulation_params['time_periods'])]



if args.heuristic == "random":
	groups = []
	for group in dynModel.parameters['seir-groups']:
		population = sum([dynModel.initialization[group][sg] for sg in ["S","E","I","R","Ia","Ips","Ims","Iss","Rq","H","ICU","D"]])
		if population > 0:
			groups.append(group)
	groups.sort()

	a_tests_vec, m_tests_vec = random_partition(dynModel, groups, max_a_tests, max_m_tests)
elif args.heuristic == "homogeneous":
	a_tests_vec, m_tests_vec = homogeneous(dynModel, max_a_tests, max_m_tests)
elif "age_group" in args.heuristic:
	a_tests_vec, m_tests_vec = all_to_one(dynModel, args.heuristic, max_a_tests, max_m_tests)
elif args.heuristic == "no_tests":
	a_tests_vec, m_tests_vec = no_tests(dynModel)
elif args.heuristic == "forecasting_heuristic":
	tolerance = 10
	max_iterations = 10
	death_value = 0.5
	a_tests_vec, m_tests_vec = forecasting_heuristic(dynModel, max_a_tests, max_m_tests, alphas_vec, [dynModel.beds for t in range(len(max_a_tests))], [dynModel.icus for t in range(len(max_a_tests))], tolerance, max_iterations, death_value, mixing_method)
#ICU CAP replaced by single value dynModel.icus
tests = {
	'a_tests_vec':a_tests_vec,
	'm_tests_vec':m_tests_vec,
}



# Run the model for the whole time range
for t in range(simulation_params['time_periods']):
	dynModel.take_time_step(m_tests_vec[t], a_tests_vec[t], alphas_vec[t])

# Print model stats
dynModel.print_stats()

# Get contacts information
n_contacts_received = dynModel.n_contacts
n_contacts_given = [{g1:{g2:n_contacts_received[t][g2][g1] for g2 in dynModel.groups} for g1 in dynModel.groups} for t in range(dynModel.time_steps)]

# Draw plots
time_axis = [i*simulation_params["dt"] for i in range(simulation_params['time_periods']+1)]

groups = dynModel.groups.keys()
groups.sort()
plt.figure(1)
for i,group in enumerate(groups):
	plt.subplot(13,len(groups),i+1)
	plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
	plt.title(group)
	plt.legend(loc='upper right')
	plt.ylim(-10000,np.max([np.max(dynModel.groups[group].S) for group in groups]))

for i,group in enumerate(groups):
	plt.subplot(13,len(groups),i+1+len(groups))
	plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
	plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
	plt.legend(loc='upper right')
	plt.ylim(-10000,np.max([max(np.max(dynModel.groups[group].E),np.max(dynModel.groups[group].I)) for group in groups]))

for i,group in enumerate(groups):
	plt.subplot(13,len(groups),i+1+len(groups)*2)
	plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
	plt.ylim(-10000,np.max([np.max(dynModel.groups[group].R) for group in groups]))
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(13,len(groups),i+1+len(groups)*3)
	plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
	plt.ylim(-10000,np.max([np.max(dynModel.groups[group].Rq) for group in groups]))
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(13,len(groups),i+1+len(groups)*4)
	plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
	plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
	plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
	plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
	plt.ylim(-10000,np.max([max(np.max(dynModel.groups[group].Ia),np.max(dynModel.groups[group].Ips),np.max(dynModel.groups[group].Ims),np.max(dynModel.groups[group].Iss)) for group in groups]))
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(13,len(groups),i+1+len(groups)*5)
	plt.plot(time_axis, dynModel.groups[group].H, label="Hospital Bed")
	plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
	plt.plot(time_axis, dynModel.groups[group].D, label="Dead")
	plt.ylim(-10000,np.max([max(np.max(dynModel.groups[group].H),np.max(dynModel.groups[group].ICU),np.max(dynModel.groups[group].D)) for group in groups]))
	plt.legend(loc='upper right')


for i,group in enumerate(groups):
	plt.subplot(13,len(groups),i+1+len(groups)*6)
	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(re_change_order(m_tests_vec)[group])+max(float(args.m_tests),float(args.a_tests))/100, label="M Tests")
	plt.plot(range(0,int(simulation_params['time_periods'])), re_change_order(a_tests_vec)[group], label="A Tests")
	plt.ylim(-max(float(args.m_tests),float(args.a_tests))/10,max(float(args.m_tests),float(args.a_tests))+max(float(args.m_tests),float(args.a_tests))/10)
	plt.legend(loc='upper right')


dic_alphas = change_order_alphas(alphas_vec)
for i,group in enumerate(groups):
	plt.subplot(13,len(groups),i+1+len(groups)*7)
	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["home"][:simulation_params['time_periods']])+0.01, label="Home")
	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["work"][:simulation_params['time_periods']])+0.01*2, label="Work")
	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["leisure"][:simulation_params['time_periods']])+0.01*3, label="Leisure")
	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["school"][:simulation_params['time_periods']])+0.01*4, label="School")
	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["other"][:simulation_params['time_periods']])+0.01*5, label="Other")
	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["transport"][:simulation_params['time_periods']])+0.01*6, label="Transport")
	plt.ylim(-0.1,1.1)
	plt.legend(loc='upper right')

plt.subplot(13,2,17)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
plt.legend(loc='upper right')

plt.subplot(13,2,18)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.legend(loc='upper right')



maximum = max([max([max([n_contacts_received[t][group][group2] for t in range(dynModel.time_steps)]) for group in groups]) for group2 in groups]) + 0.2
for i,group in enumerate(groups):
	plt.subplot(7,len(groups),i+1+len(groups)*5)
	for j,group2 in enumerate(groups):
		plt.plot(range(dynModel.time_steps), [n_contacts_received[t][group][group2] for t in range(dynModel.time_steps)], label=group2)
	if i==0:
		plt.ylabel("Contacts Received")
	plt.legend(loc='upper right')
	plt.ylim(0,maximum)

for i,group in enumerate(groups):
	plt.subplot(7,len(groups),i+1+len(groups)*6)
	for j,group2 in enumerate(groups):
		plt.plot(range(dynModel.time_steps), [n_contacts_given[t][group][group2] for t in range(dynModel.time_steps)], label=group2)
	if i==0:
		plt.ylabel("Contacts Given")
	plt.legend(loc='upper right')
	plt.ylim(0,maximum)





#
# # Draw plots
# time_axis = [i*simulation_params["dt"] for i in range(simulation_params['time_periods']+1)]
#
# groups = dynModel.groups.keys()
# plt.figure(1)
# for i,group in enumerate(groups):
# 	plt.subplot(9,len(groups),i+1)
# 	plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
# 	plt.title(group)
# 	plt.legend(loc='upper right')
# 	plt.ylim(-10000,np.max([np.max(dynModel.groups[group].S) for group in groups]))
#
# for i,group in enumerate(groups):
# 	plt.subplot(9,len(groups),i+1+len(groups))
# 	plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
# 	plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
# 	plt.legend(loc='upper right')
# 	plt.ylim(-10000,np.max([max(np.max(dynModel.groups[group].E),np.max(dynModel.groups[group].I)) for group in groups]))
#
# for i,group in enumerate(groups):
# 	plt.subplot(9,len(groups),i+1+len(groups)*2)
# 	plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
# 	plt.ylim(-10000,np.max([np.max(dynModel.groups[group].R) for group in groups]))
# 	plt.legend(loc='upper right')
#
# for i,group in enumerate(groups):
# 	plt.subplot(9,len(groups),i+1+len(groups)*3)
# 	plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
# 	plt.ylim(-10000,np.max([np.max(dynModel.groups[group].Rq) for group in groups]))
# 	plt.legend(loc='upper right')
#
# for i,group in enumerate(groups):
# 	plt.subplot(9,len(groups),i+1+len(groups)*4)
# 	plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
# 	plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
# 	plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
# 	plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
# 	plt.ylim(-10000,np.max([max(np.max(dynModel.groups[group].Ia),np.max(dynModel.groups[group].Ips),np.max(dynModel.groups[group].Ims),np.max(dynModel.groups[group].Iss)) for group in groups]))
# 	plt.legend(loc='upper right')
#
# for i,group in enumerate(groups):
# 	plt.subplot(9,len(groups),i+1+len(groups)*5)
# 	plt.plot(time_axis, dynModel.groups[group].H, label="Hospital Bed")
# 	plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
# 	plt.plot(time_axis, dynModel.groups[group].D, label="Dead")
# 	plt.ylim(-10000,np.max([max(np.max(dynModel.groups[group].H),np.max(dynModel.groups[group].ICU),np.max(dynModel.groups[group].D)) for group in groups]))
# 	plt.legend(loc='upper right')
#
#
# for i,group in enumerate(groups):
# 	plt.subplot(9,len(groups),i+1+len(groups)*6)
# 	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(re_change_order(m_tests_vec)[group])+max(args.m_tests,args.a_tests)/100, label="M Tests")
# 	plt.plot(range(0,int(simulation_params['time_periods'])), re_change_order(a_tests_vec)[group], label="A Tests")
# 	plt.ylim(-max(args.m_tests,args.a_tests)/10,max(args.m_tests,args.a_tests)+max(args.m_tests,args.a_tests)/10)
# 	plt.legend(loc='upper right')
#
#
# dic_alphas = change_order_alphas(alphas_vec)
# for i,group in enumerate(groups):
# 	plt.subplot(9,len(groups),i+1+len(groups)*7)
# 	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["home"][0:int(simulation_params['time_periods'])])+0.01, label="Home")
# 	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["work"][0:int(simulation_params['time_periods'])])+0.01*2, label="Work")
# 	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["leisure"][0:int(simulation_params['time_periods'])])+0.01*3, label="Leisure")
# 	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["school"][0:int(simulation_params['time_periods'])])+0.01*4, label="School")
# 	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["other"][0:int(simulation_params['time_periods'])])+0.01*5, label="Other")
# 	plt.plot(range(0,int(simulation_params['time_periods'])), np.array(dic_alphas[group]["transport"][0:int(simulation_params['time_periods'])])+0.01*6, label="Transport")
# 	plt.ylim(-0.1,1.1)
# 	plt.legend(loc='upper right')
#
#
# plt.subplot(9,2,17)
# #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
# plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
# #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
# plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
# plt.legend(loc='upper right')
#
# plt.subplot(9,2,18)
# #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
# plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
# #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
# plt.legend(loc='upper right')
#
#
figure = plt.gcf() # get current figure
figure.set_size_inches(7*len(groups),24)
figure.suptitle('Region: %s, Policy: %s, MTests/day: %s, ATests/day: %s, Heuristic: %s, Infected: %s, Total Deaths: %s, Total Economic Value: %s'%(simulation_params['region'],args.policy,args.m_tests,args.a_tests, args.heuristic,args.perc_infected, dynModel.get_total_deaths(), dynModel.get_total_economic_value()), fontsize=22)

if args.heuristic == "forecasting_heuristic":
	plt.savefig("results_runs/"+simulation_params['region']+"_params_"+args.policy+"_heuristic_"+args.heuristic+"_infected_"+args.perc_infected+"_cost_of_death_"+str(death_value)+"_days_"+str(simulation_params['days'])+"_mixing_"+args.mixing+".pdf")
else:
	plt.savefig("results_runs/"+simulation_params['region']+"_params_"+args.policy+"_heuristic_"+args.heuristic+"_infected_"+args.perc_infected+"_days_"+str(simulation_params['days'])+"_mixing_"+args.mixing+".pdf")
