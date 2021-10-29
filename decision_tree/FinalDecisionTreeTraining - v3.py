# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

#@title Import revelant modules and define some useful functions
from IPython import get_ipython
import os
import sys
assert sys.version_info >= (3, 5)   # Python ≥3.5 is required
import urllib

import numpy as np   # numpy for numerical linear algebra
import pandas as pd  # pandas for managing dataframes
import time 

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
# get_ipython().run_line_magic('matplotlib', 'inline')
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

import pickle

def append_dummies(data, columns):
  categories = {}  # the dictionary with categories
  data_with_dummies = data.copy() # the changed dataframe
  for col in columns:
    if col not in data.columns:
      print("WARNING. Column '{}' not among the columns in the dataframe. Skipping it.")
    elif (data[col].dtype!=int and data[col].dtype!=float):
      dummy_df = pd.get_dummies(data[col])
      categories[col] = list(dummy_df.columns)  # keep all the categories
      data_with_dummies = pd.merge( left=data_with_dummies,                                     right=dummy_df,                                    left_index=True, right_index=True,                                   how="inner", suffixes=("", "") )
  return data_with_dummies, categories


# Choose activity for tests and load the dataset

for activity in ["work", "leisure", "transport", "other", "school"]:
    print(f"Training activity: {activity}")
    # activity = "work" 
    #activity = "leisure"
    #activity = "transport"
    # activity = "other"
    #activity = "school"

    local_csv = "./datasets/dataset_training_testing_"+activity+".csv"   # name of local file where you want to store the downloaded file


    # local_csv = "./datasets/dataset_evaluating_"+activity+".csv"   # name of local file where you want to store the downloaded file


    # # %%
    # print(pd.__version__)


    # %%
    # Read the data into a `pandas` DataFrame
    print(f"Reading CSV...")
    start_read_csv_time = time.time()

    data = pd.read_csv(local_csv, index_col=None)

    end_csv_time = time.time()
    print(f"Finished reading CSV. Took {end_csv_time - start_read_csv_time} seconds.")

    # # %%
    # list(data.columns)

    # %% [markdown]
    # Count NaN's in the dataset:

    # %%
    # data.isnull().sum().sum()

    # %% [markdown]
    # Take closer look at rows with NaN's:

    # # %%
    # if data.isnull().sum().sum() > 0:
    #     # print the non-null counts for each feature
    #     data.isna().sum()
        
    #     data_NaN_rows = data[data.isna().any(axis=1)]
    #     print(data_NaN_rows.iloc[0])
    #     print(data_NaN_rows.iloc[359])

    # %% [markdown]
    # Are all the NaN's in the `econ_contacts_death_cost_ratio`?

    # %%
    # data.columns[data.isna().any()].tolist()

    # %% [markdown]
    # #### <font color=red> This is due to division of 0/0, because of `chi+vg_life` being 0.  Will need to remember this when training the trees.</font>
    # 
    # #### <font color=green>  Agreed. </font>
    # 
    # #### <font color=blue> [DI] We need to figure out HOW to resolve it. Not sure that deleting the rows is the best option.</font>
    # 
    # #### <font color=red> Leaving this check and comment in for now until we have converged on a way forward, such as removing the `econ_to_contacts_death_cost_ratio` feature altogether.</font>

    # %%
    # summary stats
    # data.describe()

    # %% [markdown]
    # # Clean up the data 
    # %% [markdown]
    # Change $\chi$ and $\chi + \text{vg_life}$ values into multiples of GDP.

    # %%
    GDP_per_capita = 37199.03
    data["chi_orig"] = data["chi"]
    data["chi"] = data["chi_orig"]/GDP_per_capita

    print(np.sort(data["chi"].unique()))

    data["chi+vg_life_orig"] = data["xi+vg_life"]
    data["chi+vg_life"] = data["chi+vg_life_orig"]/GDP_per_capita

    # print(np.sort(data["chi+vg_life"].unique()))

    # %% [markdown]
    # Add a new feature corresponding to the total population in the group without deaths. We add this as a proxy for the total population in the group that is working, which is what appears in the objective. This is easier to interpret, and DMs can compute it in practice.

    # %%
    alive_states = ['S', 'E', 'I', 'Ia', 'Ims', 'Ips', 'Iss', 'ICU', 'H', 'R', 'Rq']
    data["Nalive"] = sum(data[s] for s in alive_states )

    # %% [markdown]
    # # Some Brief EDA

    # %%
    # Summary stats
    # data.describe()

    # %% [markdown]
    # # Train a Decision Tree

    # %%
    from sklearn import tree
    from sklearn.metrics import mean_squared_error

    # %% [markdown]
    # ### Encode age group as numeric feature
    # Add another column with age encoded as a numeric feature

    # %%
    # ordinal encoding - two options:
    #  1) replace with midpoint of closed interval (e.g., 5, 15, 25, etc; for 80+, will use 85)
    #  2) replace age bracket with upper limit (for the 80-plus, will use 89)
    use_midpoint  = True

    data["age"] = data["age_group"]

    unique_age_groups = data["age"].unique()
    for v in unique_age_groups:

        all_splits = v.split("_")
        lower_lim = np.int(all_splits[-2])
        upper_lim = all_splits[-1]

        
        
        if upper_lim=="plus":
            upper_lim = 89
        else:
            upper_lim = np.int(upper_lim)


        if use_midpoint:
            data.loc[ data["age"]==v, "age" ] = (lower_lim + upper_lim + 1)/2
            
            
        else:
            data.loc[ data["age"]==v, "age" ] = upper_lim

        # convert entire column for age into integer
    data = data.astype({'age':int})


    # %%
    # data.head(9)


    # %%
    # list(data.columns)

    # %% [markdown]
    # ### Choose the variables to train the tree with; set filters; Train the tree, Display, Save to a File; Implement Cross-Validation
    # %% [markdown]
    # ### Comments on features to keep in the trees
    # We remove the following features:
    #  - $\chi, v_g^{\mbox{life}}$ : we only retain $\chi + v_g^{\mbox{life}}$, since that is the only thing appearing in the model
    #  - we eliminate $p_{ss}$ since it is the sum of $p^H$ and $p^{ICU}$
    #  - 'econ_gradient_not_normalized': remove and only retain the other one
    # %% [markdown]
    # ### Our plan will be to run a few different versions of trees:
    #  1. ALL: A tree with ALL the features (after the cleaning above). The point of this exercise is more to show that econ_to_contacts  is the most relevant feature, and also to see how good this is relative to ROLD.
    #  2. SIMPLE: A tree that only uses simple features, i.e. things that regulators are very likely to look at (see list below).
    #  3. SIMPLE_PLUS: A tree with the same simple features as #2 and the `econ_to_contacts_ratio`. 
    # %% [markdown]
    # 

    # %%
    # features for complete tree (#1 in list)
    # all_features = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    # 'chi+vg_life', 'p_H', 'p_ICU', 'p_D', 'econ_gradient_normalized', 'I_perc', 'R_perc', 'H_perc', 'ICU_utilization',
    # 'contacts_received', 'contacts_received_into_deaths', 'infections_received', 'deaths_received', 'deaths_received_econ',
    # 'contacts_given', 'contacts_given_into_deaths', 'infections_given', 'deaths_given', 'deaths_given_econ',
    # 'econ_contacts_ratio', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    # 'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'S', 'E', 'I', 'Ia', 'Ims', 'Ips',
    # 'Iss', 'ICU', 'H', 'R', 'Rq', 'D', 'age', 'Nalive']

    # features  for the simple tree
    simple_features = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_D', 'econ_gradient_normalized',  'R_perc', 'ICU_utilization',
    'contacts_received', 'contacts_given', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'age']

    original_plus = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_H', 'p_ICU', 'p_D', 'econ_gradient_normalized', 'I_perc', 'R_perc', 'H_perc', 'ICU_utilization',
    'contacts_received', 'contacts_received_into_deaths', 'infections_received', 'deaths_received', 'deaths_received_econ',
    'contacts_given', 'contacts_given_into_deaths', 'infections_given', 'deaths_given', 'deaths_given_econ',
    'econ_contacts_ratio', 'age'] 

    original = ['t', 'icu_capacity', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_H', 'p_ICU', 'p_D', 'econ_gradient_normalized', 'I_perc', 'R_perc', 'H_perc', 'ICU_utilization',
    'contacts_received', 'contacts_received_into_deaths', 'infections_received', 'deaths_received', 'deaths_received_econ',
    'contacts_given', 'contacts_given_into_deaths', 'infections_given', 'deaths_given', 'deaths_given_econ',
    'econ_contacts_ratio', 'age']

    engineered_econ_to_contacts_ratio_plus = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_D', 'R_perc',  'ICU_utilization', 
    'econ_contacts_ratio', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'age']

    engineered_econ_to_contacts_ratio_minus = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    'chi+vg_life', 'p_D', 'R_perc',  'ICU_utilization', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'age']

    # engineered_1 = ['t', 'icu_capacity', 'theta_schooling', 'beta', 'alpha_1_2', 'nu_other_act', 'nu_work',
    # 'chi+vg_life', 'p_D', 'econ_gradient_normalized', 'R_perc',  'ICU_utilization',
    # 'contacts_received', 'contacts_received_into_deaths', 'infections_received', 'deaths_received', 'deaths_received_econ',
    # 'contacts_given', 'contacts_given_into_deaths', 'infections_given', 'deaths_given', 'deaths_given_econ',
    # 'econ_contacts_ratio', 'ICU_new_daily_admissions', 'ICU_seven_day_avg_daily_admissions',
    # 'Infections_new_rate_over_seven_days', 'Infections_new_rate_over_seven_days_elderly', 'age']

    # features  for the simple tree and econ to contacts ratio
    # simple_plus_features = simple_features + ['econ_contacts_ratio']
    simple_plus_features = simple_features + ['econ_contacts_ratio']


    # %%
    #  Building the tree

    #@markdown (Optional) Select other features and associated values to use for filtering:
    #@param ['None', 'age', 'lockdown', 'econ_gradient', 'I_perc', 'R_perc', 'H_perc', 'ICU_utilization', \
    # 'xi+cost_death', 'xi', 'cost_death', 'contacts_received', 'contacts_received(deaths)', 'infections_received', 
    # 'deaths_received','deaths_econ_received', 'contacts_given', 'contacts_given(deaths)','infections_given', 
    # 'deaths_given', 'deaths_econ_given', 'death_prob','t']
    filter_feature_1 = "None" 
    #filter_feature_1 = "xi" 
    filter_feature_1_value = 1859951.5 #@param {type:"raw"}

    filter_feature_2 = "None" 
    #@param ['None', 'age', 'lockdown', 'econ_gradient', 'I_perc', 'R_perc', 'H_perc', 'ICU_utilization', \
    # 'xi+cost_death', 'xi', 'cost_death', 'contacts_received', 'contacts_received(deaths)', 'infections_received', \
    # 'deaths_received','deaths_econ_received', 'contacts_given', 'contacts_given(deaths)','infections_given', \
    # 'deaths_given', 'deaths_econ_given', 'death_prob','t']
    filter_feature_2_value = 0 #@param {type:"raw"}

    #filter_feature_2 = "xi" #@param ['None', 'age', 'lockdown', 'econ_gradient', 'I_perc', 'R_perc', 'H_perc', \
    # 'ICU_utilization', 'xi+cost_death', 'xi', 'cost_death', 'contacts_received', 'contacts_received(deaths)', \
    # 'infections_received', 'deaths_received','deaths_econ_received', 'contacts_given', 'contacts_given(deaths)',\
    # 'infections_given', 'deaths_given', 'deaths_econ_given', 'death_prob','t']
    #filter_feature_2_value = 1859951.5 #@param {type:"raw"}

    filter_feature_3 = 'None' #@param ['None', 'age', 'lockdown', 'econ_gradient', 'I_perc', 'R_perc', 'H_perc', \
    # 'ICU_utilization', 'xi+cost_death', 'xi', 'cost_death', 'contacts_received', 'contacts_received(deaths)', \
    # 'infections_received', 'deaths_received','deaths_econ_received', 'contacts_given', 'contacts_given(deaths)',\
    # 'infections_given', 'deaths_given', 'deaths_econ_given', 'death_prob','t']
    filter_feature_3_value_min = 0 #@param {type:"raw"}
    filter_feature_3_value_max = 200

    filter_feature_4 = 'None' #@param ['None', 'age', 'lockdown', 'econ_gradient', 'I_perc', 'R_perc', 'H_perc', \
    # 'ICU_utilization', 'xi+cost_death', 'xi', 'cost_death', 'contacts_received', 'contacts_received(deaths)', \
    # 'infections_received', 'deaths_received','deaths_econ_received', 'contacts_given', 'contacts_given(deaths)', \
    # 'infections_given', 'deaths_given', 'deaths_econ_given', 'death_prob','t']
    filter_feature_4_value_min = 10 #@param {type:"raw"}
    filter_feature_4_value_max = 70 #@param {type:"raw"}

    # CREATE A SUITABLE TRAINING SET
    pydot_graph = {}  #dictionary with graphs for each selected activity

    for var_title in ["SIMPLE", "SIMPLE_PLUS", "ORIGINAL_PLUS", "ORIGINAL", "ENGINEERED_1_PLUS", "ENGINEERED_1_MINUS"]:
        
        
        # SELECT THE FEATURES
        # if var_title == "ALL":
        #     vars_to_use = all_features
        # #var_title = "ALL"
        
        if var_title == "SIMPLE":
            vars_to_use = simple_features
        #var_title = "SIMPLE"
        
        if var_title == "SIMPLE_PLUS":
            vars_to_use = simple_plus_features
        # var_title = "SIMPLE_PLUS"

        if var_title == "ORIGINAL_PLUS":
            vars_to_use = original_plus
        # var_title = "SIMPLE_PLUS"

        if var_title == "ORIGINAL":
            vars_to_use = original
        # var_title = "SIMPLE_PLUS"

        if var_title == "ENGINEERED_1_PLUS":
            vars_to_use = engineered_econ_to_contacts_ratio_plus
        # var_title = "SIMPLE_PLUS"

        if var_title == "ENGINEERED_1_MINUS":
            vars_to_use = engineered_econ_to_contacts_ratio_minus
        # var_title = "SIMPLE_PLUS"

        print("Building tree for {} activity, using {} features".format(activity,var_title))

        # # FILTER BASED ON CONDITIONS
        # filter = [True]*len(data)
        # if filter_feature_1!="None":
        #     title += "_{}_{}".format(filter_feature_1,filter_feature_1_value)
        #     filter = filter & (data[filter_feature_1] == filter_feature_1_value)
        # if filter_feature_2!="None":
        #     title += "_{}_{}".format(filter_feature_2,filter_feature_2_value)
        #     filter = filter & (data[filter_feature_2] == filter_feature_2_value)
        # if filter_feature_3!="None":
        #     filter = filter & (data[filter_feature_3] >= filter_feature_3_value_min) & (data[filter_feature_3] <= filter_feature_3_value_max)
        #     title += "_{}_min_{}_max_{}".format(filter_feature_3,filter_feature_3_value_min,filter_feature_3_value_max)
        # if filter_feature_4!="None":
        #     filter = filter & (data[filter_feature_4] >= filter_feature_4_value_min) & (data[filter_feature_4] <= filter_feature_4_value_max)
        #     title += "_{}_min_{}_max_{}".format(filter_feature_4,filter_feature_4_value_min,filter_feature_4_value_max)

        print(f"finished filtering data")

        # CREATE THE DATASET
        # filtered_data = data[filter]  # only rows for the correct activity

        print(f"created dataset")

        # remove all rows with missing data (for now...)
        if( data.isna().sum().sum() > 0 ):
            data = data.dropna()
        print("Applying the filter for '{}' results in {} unique records.".format(activity,len(data)))

        # select the features
        X_train = data.loc[:, vars_to_use ]
        y_train = data.loc[:, "lockdown"]
        

        # TRAIN THE TREE
        # USE CROSS VALIDATION TO SEARCH FOR BEST max_depth
        from sklearn.model_selection import GridSearchCV
        param_grid = {'max_depth': np.arange(7,8,1), 'random_state' : [123]}

        # Use random_state to set the seed in order to avoid instabilities across different runs due to randomization in the folding
        dt_model = tree.DecisionTreeRegressor(criterion='mse',random_state=0)

        print("defined grid search")
        grid_search = GridSearchCV(dt_model, param_grid, cv=2, scoring='neg_mean_squared_error', n_jobs=2, refit=True, return_train_score=True)

        # fit using CV
        time_pre_fitting = time.time()
        print(f"fitting using CV")
        grid_search.fit(X_train, y_train)

        time_post_fitting = time.time()

        print(f"total time fitting: {(time_post_fitting - time_pre_fitting)/60} minutes.")

        # get the best estimator
        dt_model = grid_search.best_estimator_   
        max_depth = dt_model.get_params()['max_depth']

        print("Optimal depth: {}".format(max_depth))

        # IF YOU WANT TO DISPLAY THE TREE UNCOMMENT THE NEXT SECTION
        #dot_data = tree.export_graphviz(dt_model, max_depth=opt_max_depth, \
        #                              feature_names=feature_names, \
        #                              class_names=None, label='all', filled=True, impurity=False, node_ids=False, \
        #                              rounded=True, rotate=True, proportion=False, precision=4)
        #graph = graphviz.Source(dot_data)  
        #display(graph)

        # WE NEED TO CLEAN UP THE NAMES BELOW

        # WRITE TREE TO DOT FILE
        feature_names = list( X_train.columns )
        if 'chi' in feature_names:
            feature_names[feature_names.index('chi')] = chr(967)
        if 'chi+vg_life' in feature_names:
            feature_names[feature_names.index('chi+vg_life')] = chr(967)+' '+'+'+' '+'vg-life'    
        if 't' in feature_names:
            feature_names[feature_names.index('t')] = 'time'
        if 'econ_contacts_ratio' in feature_names:
            feature_names[feature_names.index('econ_contacts_ratio')] = 'econ-to-contacts-ratio'
        if 'econ_contacts_death_cost_ratio' in feature_names:
            feature_names[feature_names.index('econ_contacts_death_cost_ratio')] = 'econ-to-contacts-deathcost-ratio'
        if 'econ_gradient_trunc' in feature_names:
            feature_names[feature_names.index('econ_gradient_trunc')] = 'econ-gradient'
        if 'R_perc' in feature_names:
            feature_names[feature_names.index('R_perc')] = 'R-perc'
        if 'contacts_received' in feature_names:
            feature_names[feature_names.index('contacts_received')] = 'contacts-received'
        if 'deaths_given_econ' in feature_names:
            feature_names[feature_names.index('deaths_given_econ')] = 'deaths-given-econ'

        dot_data = tree.export_graphviz(dt_model, max_depth=max_depth,                               feature_names=feature_names,                               class_names=None, label='all', filled=True, impurity=False, node_ids=False,                               rounded=True, rotate=True, proportion=False, precision=4)

        # WRITE TO A PDF FILE AND DOWNLOAD IT
        title = activity

        # adjust the file name to reflect the tree depth and which vars are used 
        title += "_Dep_" + str(max_depth)
        title += '_variables_' + var_title

        # calculate MSE manually and check against the printed value
        y_hat = dt_model.predict(X_train)
        mse = sklearn.metrics.mean_squared_error(y_hat, y_train)              

        # read in dot file and add a node with the correct MSE (for the prediction using the tree)
        pydot_graph[activity] = pydotplus.graph_from_dot_data(dot_data)

        # change any nodes with "age <=" into "age <"
        for n in pydot_graph[activity].get_node_list():
            label = n.get_label()

            if label is not None:
                #print("Original label: {}".format(label))

                # change the precision for the logical condition
                if label.find('<=')>0:
                    # not a leaf node
                    logical_cond_thresh = (label.split("<= ")[1]).split('\\')[0]  # what comes after "<= " and before "\"
                    logical_cond_thresh_val = np.float(logical_cond_thresh)  # the value just before the \
                    label = label.replace(logical_cond_thresh, "{:.1f}".format(logical_cond_thresh_val))

                # change the formatting for samples (add ,)
                samples_str = (label.split("samples = ")[1]).split('\\')[0]  # what comes after "samples = " and before "\"
                samples_val = np.float(samples_str)  # the value just before the \
                label = label.replace(samples_str, "{:,.0f}".format(samples_val))

                node_value_str = label.split("value = ")[1][0:-1]  # what comes after "value =" in the node label
                node_value = np.float(node_value_str)
                #print("Node value: {}".format(node_value_str))

                offset = 0.0
                n.set_fillcolor("0.0 0.0 "+str(offset + node_value*(1-offset)))

                if node_value<0.5:
                    # for nodes that are sufficiently dark, make the fonts white
                    n.set_fontcolor('white')

                label = label.replace("value","activity level")
                label = label.replace(node_value_str,"{:.2f}%".format(node_value*100))

                #in case we need to change the label for age...
                if label.find("age <=")>0:
                    label.replace("age <=","age <")

                # print("New label {}\n".format(label))
                n.set_label(label)

        # calculate MSE
        mse_string = "RMSE: {:.5f}".format(np.sqrt(mse))
        print(mse_string)

        show_rmse = True
        if show_rmse:
            title_node = pydotplus.Node(label='Decision tree for "'+activity+'"\n'+mse_string,                                 fontcolor="black", fillcolor="white", fontsize=18, style="bold")
        else:
            title_node = pydotplus.Node(label='Decision tree for "'+activity+'"',                                 fontcolor="black", fillcolor="white", fontsize=18, style="bold")

        pydot_graph[activity].add_node(title_node)

        folder_pdf = "./trees_pdf/"
        file_name = folder_pdf + title + ".pdf"
        pydot_graph[activity].write_pdf(file_name)

        folder_pickle = "./trees_pickles/"
        with open(folder_pickle + title + ".pickle", "wb") as pickle_file:
            pickle.dump(dt_model, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


        # DOWNLOAD THE PDF -- uncomment if in COLAB
        # files.download(file_name)


# %%



# %%



