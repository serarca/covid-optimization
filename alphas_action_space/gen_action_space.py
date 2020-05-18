import pickle
import pprint
import yaml
pp = pprint.PrettyPrinter(indent=4)
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()


# Import data
data_dict = pickle.load( open( "../data/data_dict.p", "rb" ) )

action_space = {}


for group in data_dict['lockdown_patterns']:
	action_space[group] = {}
	for activity in data_dict['lockdown_patterns'][group]:
		for lockdown,value in data_dict['lockdown_patterns'][group][activity].iteritems():
			if lockdown in action_space[group]:
				pass
			else:
				action_space[group][lockdown] = {}
			action_space[group][lockdown][activity] = value



with open('default.yaml', 'w') as file:
    yaml.dump(action_space, file)