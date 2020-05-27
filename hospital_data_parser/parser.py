import numpy as np
import csv
import pandas as pd

### Define instance object
class CovidParser:
	def __init__(self, filePath):
		self.filePath = filePath;
		self.df_master = pd.read_csv(self.filePath, sep=';')

def filterDataFrame(df, dptCodes):
	# select only ile de france dpts
	df = df[df['reg'].isin(dptCodes)]
	
	# remove "0" age entries, which are an aggregation accross all age buckets
	df = df[df['cl_age90'] > 0]
	
	return df
	
def createDFsbyAgeGroup(df_master, ageBuckets, ageGroupStrings):
	dfs_by_age_dict = {}
	
	# split by age buckets
	for index in range(0, len(ageBuckets) - 1):
		if index < 8:
			dfs_by_age_dict[ageGroupStrings[index]] = df_master[df_master['cl_age90'] == ageBuckets[index]]
		else:
			dfs_by_age_dict[ageGroupStrings[index]] = df_master[df_master['cl_age90'].isin([ageBuckets[index], ageBuckets[index+1]])]
		# aggregate all idf departments into a single row
		aggregation_functions = {'reg': 'first', 'cl_age90': 'first', 'jour': 'first', 'hosp': 'sum', 'rea': 'sum', 'rad': 'sum', 'dc': 'sum',}
		dfs_by_age_dict[ageGroupStrings[index]] = dfs_by_age_dict[ageGroupStrings[index]].groupby(dfs_by_age_dict[ageGroupStrings[index]]['jour']).aggregate(aggregation_functions)
	return dfs_by_age_dict
