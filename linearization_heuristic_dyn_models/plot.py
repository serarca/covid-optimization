from inspect import getsourcefile
import os.path
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
sys.path.insert(0, current_dir+"/..")

parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
sys.path.insert(0, parent_dir+"/heuristics")

from linearization import *
from heuristics import *

import pickle
from group import SEIR_group, DynamicalModel

def unpickle_plot_and_print_results(file):
    ##############################################################################
    # PLOT AND PRINT RESULTS FROM THE DYNAMICAL MODEL SIMULATED USING THE SIMULATION PARAMS


    dynModel = pickle.load(open(file, 'rb'))



    T = dynModel.time_steps

    K_mtest = dynModel.parameters['global-parameters']['C_mtest']
    K_atest = dynModel.parameters['global-parameters']['C_atest']

    # Retrieve optimal lockdown decisions
    # Express as dictionary where given an age group, an activity key corresponds to an np.array of length T.
    # That array holds the lockdown decisions for that age group and that activity used in the simulation of dynModel.
    lockdowns_sim = {}
    for n in dynModel.groups:
        lockdowns_sim[n] = {}
        for act in dynModel.lockdown_controls[0][n]:
            lockdowns_sim[n][act] = np.zeros(T)
            for t in range(T):
                lockdowns_sim[n][act][t] = dynModel.lockdown_controls[t][n][act]

    # Retrieve simulated testing decisions
    m_tests_sim = {}
    for n in dynModel.groups:
        m_tests_sim[n] = np.zeros(T)
        for t in range(T):
            m_tests_sim[n][t] = dynModel.m_tests_controls[t][n]

    a_tests_sim = {}
    for n in dynModel.groups:
        a_tests_sim[n] = np.zeros(T)
        for t in range(T):
            a_tests_sim[n][t] = dynModel.a_tests_controls[t][n]


    # # Retrieve simulated bouncing decisions

    B_H_sim = {}
    for n,g in dynModel.groups.items():
        B_H_sim[n] = np.zeros(T-1)
        for t in range(T-1):
            B_H_sim[n][t] = g.B_H[t]

    B_ICU_sim = {}
    for n,g in dynModel.groups.items():
        B_ICU_sim[n] = np.zeros(T-1)
        for t in range(T-1):
            B_ICU_sim[n][t] = g.B_ICU[t]

    print("Deaths through the end of the horizon:", sum([dynModel.deaths[t] for t in range(0,T+1) if t!=0]))
    print("Economic output through the end of the horizon:", sum([dynModel.economic_values[t] for t in range(0,T+1) if t!=0]))



    # Plotting
    time_axis = [i*dynModel.dt for i in range(T+1)]
    time_axis_controls = [i*dynModel.dt for i in range(T)]

    groups = dynModel.groups.keys()
    groups = sorted(groups)
    fig = plt.figure()
    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1)
        plt.plot(time_axis, dynModel.groups[group].S, label="Susceptible")
        plt.title(group)
        plt.legend(loc='upper right')
        plt.ylim(-1,np.max([np.max(dynModel.groups[group].S) for group in groups]))

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups))
        plt.plot(time_axis, dynModel.groups[group].E, label="Exposed")
        plt.plot(time_axis, dynModel.groups[group].I, label="Infected")
        plt.legend(loc='upper right')
        plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].E),np.max(dynModel.groups[group].I)) for group in groups]))

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*2)
        plt.plot(time_axis, dynModel.groups[group].R, label="Recovered")
        plt.ylim(-1,np.max([np.max(dynModel.groups[group].R) for group in groups]))
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*3)
        plt.plot(time_axis, dynModel.groups[group].Rq, label="Recovered Q")
        plt.ylim(-1,np.max([np.max(dynModel.groups[group].Rq) for group in groups]))
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*4)
        plt.plot(time_axis, dynModel.groups[group].Ia, label="Infected A-Q")
        plt.plot(time_axis, dynModel.groups[group].Ips, label="Infected PS-Q")
        plt.plot(time_axis, dynModel.groups[group].Ims, label="Infected MS-Q")
        plt.plot(time_axis, dynModel.groups[group].Iss, label="Infected SS-Q")
        plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].Ia),np.max(dynModel.groups[group].Ips),np.max(dynModel.groups[group].Ims),np.max(dynModel.groups[group].Iss)) for group in groups]))
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*5)
        plt.plot(time_axis, dynModel.groups[group].H, label="Hospital")
        plt.plot(time_axis, dynModel.groups[group].ICU, label="ICU")
        plt.plot(time_axis, dynModel.groups[group].D, label="Dead")
        plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].H),np.max(dynModel.groups[group].ICU),np.max(dynModel.groups[group].D)) for group in groups]))
        plt.legend(loc='upper right')


    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*6)
        plt.plot(time_axis_controls,m_tests_sim[group], label = "M tests")
        plt.plot(time_axis_controls,a_tests_sim[group], label = "A tests")
        # plt.plot(range(0,int(simulation_params['time_periods'])),
        # np.array(re_change_order(m_tests_vec)[group])+max(float(args.m_tests),float(args.a_tests))/100, label="M Tests")
        # plt.plot(range(0,int(simulation_params['time_periods'])), re_change_order(a_tests_vec)[group], label="A Tests")
        plt.ylim(-max(float(K_mtest),float(K_atest))/10,max(float(K_mtest),float(K_atest))+max(float(K_mtest),float(K_atest))/10)
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*7)
        plt.plot(time_axis_controls, lockdowns_sim[group]["home"]+0.01, label="Home")
        plt.plot(time_axis_controls, lockdowns_sim[group]["work"]+0.01*2, label="Work")
        plt.plot(time_axis_controls, lockdowns_sim[group]["transport"]+0.01*3, label="Transport")
        plt.plot(time_axis_controls, lockdowns_sim[group]["school"]+0.01*4, label="School")
        plt.plot(time_axis_controls, lockdowns_sim[group]["leisure"]+0.01*5, label="Leisure")
        plt.plot(time_axis_controls, lockdowns_sim[group]["other"]+0.01*6, label="Other")
        plt.ylim(-0.1,1.1)
        plt.legend(loc='upper right')

    for i,group in enumerate(groups):
        plt.subplot(13,len(groups),i+1+len(groups)*8)
        plt.plot(time_axis_controls, dynModel.groups[group].B_H, label="Bounced H")
        plt.plot(time_axis_controls, dynModel.groups[group].B_ICU, label="Bounced ICU")
        plt.ylim(-1,np.max([max(np.max(dynModel.groups[group].B_H),np.max(dynModel.groups[group].B_ICU)) for group in groups]))
        plt.legend(loc='upper right')

    plt.subplot(13,2,19)
    #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
    plt.plot(time_axis, [sum([dynModel.groups[group].ICU[i] for group in groups]) for i in range(len(time_axis))], label="Total ICUs")
    #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
    plt.axhline(y=dynModel.icus, color='g', linestyle='dashed', label= "ICU Capacity")
    plt.legend(loc='upper right')

    plt.subplot(13,2,20)
    #plt.plot(time_axis, [sum([dynModel.groups[group].H[i] for group in groups]) for i in range(len(time_axis))], label="Total Hospital Beds")
    plt.plot(time_axis, [sum([dynModel.groups[group].D[i] for group in groups]) for i in range(len(time_axis))], label="Total Deaths")
    #plt.axhline(y=parameters['global-parameters']['C_H'], color='r', linestyle='dashed', label= "Hospital Capacity")
    plt.legend(loc='upper right')

    figure = plt.gcf() # get current figure
    figure.set_size_inches(7*len(groups),24)
    figure.suptitle('Total Days: %s, M-test daily capacity: %s, A-test daily capacity: %s, Mixing: %s'%(T,K_mtest,K_atest,dynModel.mixing_method["name"]), fontsize=22)
    plt.savefig(file[:-2]+".pdf")


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--i", help="Input file.")

args = parser.parse_args()

in_file = args.i

unpickle_plot_and_print_results(in_file)
