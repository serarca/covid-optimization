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
parentdir = os.path.dirname(current_dir)
sys.path.insert(0,parentdir) 
sys.path.insert(0, parentdir+"/heuristics")
sys.path.insert(0, parentdir+"/heuristics/LP-Models")


from group import SEIR_group, DynamicalModel
from heuristics import *
import math
import pprint
import numpy as np



policy_file = "static_infected_10_m_tests_1000_a_tests_1000_bouncing_True"

with open("./results/"+policy_file+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    policy = yaml.load(file, Loader=yaml.FullLoader)


# Read group parameters
with open("../parameters/"+policy["metadata"]["region"]+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    universe_params = yaml.load(file, Loader=yaml.FullLoader)


dynModel = DynamicalModel(universe_params, policy["initialization"], policy["metadata"]['dt'], policy["metadata"]['time_periods'], policy["mixing_method"])

for t in range(policy["metadata"]['time_periods']):
	if policy["B_ICU_perc"] is not False:
		dynModel.take_time_step(policy["m_tests"][t], policy["a_tests"][t], policy["alphas"][t], B_ICU_perc=policy["B_ICU_perc"][t])
	else:
		dynModel.take_time_step(policy["m_tests"][t], policy["a_tests"][t], policy["alphas"][t])

print((dynModel.get_total_reward()-policy["optimal_value"])/dynModel.get_total_reward())


# Draw plots
time_axis = [i*policy["metadata"]['dt'] for i in range(policy["metadata"]['time_periods']+1)]

groups = dynModel.groups.keys()
groups = sorted(groups)
plt.figure(1)
for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1)
	plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
	plt.title(group)
	plt.legend(loc='upper right')
	plt.ylim(-1,np.max([np.max(dynModel.groups[group].S) for group in groups]))

for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1+len(groups))
	plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
	plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
	plt.legend(loc='upper right')
	plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].E),np.max(dynModel.groups[group].I)) for group in groups]))

for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1+len(groups)*2)
	plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
	plt.ylim(-1,np.max([np.max(dynModel.groups[group].R) for group in groups]))
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1+len(groups)*3)
	plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
	plt.ylim(-1,np.max([np.max(dynModel.groups[group].Rq) for group in groups]))
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1+len(groups)*4)
	plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
	plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
	plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
	plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
	plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].Ia),np.max(dynModel.groups[group].Ips),np.max(dynModel.groups[group].Ims),np.max(dynModel.groups[group].Iss)) for group in groups]))
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1+len(groups)*5)
	plt.plot(time_axis, dynModel.groups[group].H, label="Hospital Bed")
	plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
	plt.plot(time_axis, dynModel.groups[group].D, label="Deaths")
	plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].H),np.max(dynModel.groups[group].ICU),np.max(dynModel.groups[group].D)) for group in groups]))
	plt.legend(loc='upper right')


for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1+len(groups)*6)
	plt.plot(range(0,int(policy["metadata"]['time_periods'])),
		np.array(re_change_order(policy["m_tests"])[group])+max(policy["max_m_tests"],policy["max_a_tests"])/100, label="M Tests")
	plt.plot(range(0,int(policy["metadata"]['time_periods'])), re_change_order(policy["a_tests"])[group], label="A Tests")
	plt.ylim(-max(policy["max_m_tests"],policy["max_a_tests"])/10,max(policy["max_m_tests"],policy["max_a_tests"])+max(policy["max_m_tests"],policy["max_a_tests"])/10)
	plt.legend(loc='upper right')


dic_alphas = change_order_alphas(policy["alphas"])
for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1+len(groups)*7)
	plt.plot(range(policy["metadata"]['time_periods']), np.array(dic_alphas[group]["home"][:policy["metadata"]['time_periods']])+0.01, label="Home")
	plt.plot(range(policy["metadata"]['time_periods']), np.array(dic_alphas[group]["work"][:policy["metadata"]['time_periods']])+0.01*2, label="Work")
	plt.plot(range(policy["metadata"]['time_periods']), np.array(dic_alphas[group]["leisure"][:policy["metadata"]['time_periods']])+0.01*3, label="Leisure")
	plt.plot(range(policy["metadata"]['time_periods']), np.array(dic_alphas[group]["school"][:policy["metadata"]['time_periods']])+0.01*4, label="School")
	plt.plot(range(policy["metadata"]['time_periods']), np.array(dic_alphas[group]["other"][:policy["metadata"]['time_periods']])+0.01*5, label="Other")
	plt.plot(range(policy["metadata"]['time_periods']), np.array(dic_alphas[group]["transport"][:policy["metadata"]['time_periods']])+0.01*6, label="Transport")
	plt.ylim(-0.1,1.1)
	plt.legend(loc='upper right')

for i,group in enumerate(groups):
	plt.subplot(10,len(groups),i+1+len(groups)*8)
	plt.plot(range(0,int(policy["metadata"]['time_periods'])),
		np.array(re_change_order(policy["B_ICU_perc"])[group]), label="ICU Bouncing Percentage")
	plt.ylim(-1/10,1.0 +1/10)
	plt.legend(loc='upper right')





plt.subplot(10,2,19)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
plt.legend(loc='upper right')

plt.subplot(10,2,20)
#plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
#plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
plt.legend(loc='upper right')




figure = plt.gcf() # get current figure
figure.set_size_inches(7*len(groups),24)


plt.savefig("Results/"+policy_file+".pdf")

