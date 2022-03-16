
# coding: utf-8

# In[1]:


# %matplotlib inline


import yaml
from inspect import getsourcefile
import os.path
import os
os.system('module load math')
os.system('module load python/3.6.1')
os.system('module load gurobipy3')
os.system('module load viz')
os.system('module load py-matplotlib/3.2.1_py36')
os.system('module load py-pandas/0.23.0_py36')
os.system('module load py-scipy/1.4.1_py36')
os.system('module load py-numpy/1.19.2_py36')

import numpy as np
import sys
import matplotlib
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from io import StringIO
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import datetime as DT

import seaborn as sns
# import gurobipy
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = os.path.dirname(current_dir)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.insert(0, parent_parent_dir+"/heuristics")

print(parent_parent_dir+"/heuristics")

# parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_parent_dir)


from group import SEIR_group, DynamicalModel
import math
import pprint
import pandas as pd
import pickle
# import numpy as np
from copy import deepcopy

depth = 10

perm_scores_mean = pd.read_csv(f"perm_scores_mean_all_activities_Dep_{depth}_variables_ALL_2.csv")


print(perm_scores_mean)
perm_scores_mean = perm_scores_mean.sort_values(by=["perm_score_mean"], ascending=False)

perm_scores_mean = perm_scores_mean[perm_scores_mean["perm_score_mean"] > 5e-4]

print(perm_scores_mean)

y_pos = np.arange(len(perm_scores_mean['features']))

fig, ax = plt.subplots()
# figsize=(5, 3)
# ax.barh(y_pos, perm_scores_mean['perm_score_mean'],  height=0.4, align='center', xerr=perm_scores_mean['perm_score_std'])

ax.barh(y_pos, perm_scores_mean['perm_score_mean'],  align='center', xerr=perm_scores_mean['perm_score_std'])

ax.set_yticks(y_pos)
ax.set_yticklabels(perm_scores_mean["features"])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Permutation Feature Importance')
# ax.set_ylim([, 6])
# ax.set_title('Permutation scores for all features')

plt.show()
fig.savefig(f"perm_scores_plot_dep_{depth}.pdf",bbox_inches='tight')
# ,bbox_inches='tight'

