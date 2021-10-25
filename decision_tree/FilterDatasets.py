import os
import sys
assert sys.version_info >= (3, 5)   # Python ≥3.5 is required
import urllib

import numpy as np   # numpy for numerical linear algebra
import pandas as pd  # pandas for managing dataframes

#import tensorflow as tf
#from tensorflow.keras import layers

import sklearn
from sklearn import tree
assert sklearn.__version__ >= "0.20"  # Scikit-Learn version ≥0.20 required
import sklearn.metrics as metrics

# adjust reporting in pandas: max 10 rows and two-digit precision
#pd.options.display.max_rows = 10
pd.options.display.float_format = "{:,.5f}".format

# import matplotlib and pyplot: critical packages for plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt
# Make sure Matplotlib runs inline, for nice figures

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# install the latest version of seaborn for nicer graphics
# Uncomment below line to install the package, then commment out again. Same for all pip commands in the sequel.
#!pip install seaborn==0.11.0  &> /dev/null
import seaborn as sns

from IPython.core.display import display, HTML
import base64

# install graphviz for visualizing decision trees
# !pip install graphviz
import graphviz

# install pdpbox for partial dependency visualization
# !pip install pdpbox &> /dev/null
from pdpbox import info_plots, pdp

# Ignore useless some warnings
import warnings
warnings.simplefilter(action="ignore")

# pydotplus for saving a dot file to PDF
# !pip install pydotplus
import pydotplus

for activity in ["work" , "leisure", "transport", "other", "school"]:
    local_csv = "./datasets/dataset_"+activity+".csv"   # name of local file where you want to store the downloaded file
    training_testing_local_csv = "./datasets/dataset_training_testing_"+activity+".csv" 
    evaluating_local_csv = "./datasets/dataset_evaluating_"+activity+".csv" 

    data = pd.read_csv(local_csv, index_col=None)

    data_training_testing = data.iloc[np.r_[0:31387500, 32677830:len(data.index)]]
    data_training_testing.to_csv(training_testing_local_csv)

    data_evaluating = data.iloc[np.r_[31387500:32677830]]
    data_evaluating.to_csv(evaluating_local_csv)

