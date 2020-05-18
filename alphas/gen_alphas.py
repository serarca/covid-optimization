import pickle
import pprint
import yaml
pp = pprint.PrettyPrinter(indent=4)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-lockdown", "--lockdown", help="Region")
args = parser.parse_args()

lockdown = int(args.lockdown)

# Import data
data_dict = pickle.load( open( "../data/data_dict.p", "rb" ) )

alphas = {}

for group in data_dict['lockdown_patterns']:
	alphas[group] = {}
	for activity in data_dict['lockdown_patterns'][group]:
		alphas[group][activity] = float(
			data_dict['lockdown_patterns'][group][activity][lockdown] if (group!="age_group_6" or lockdown<3) else data_dict['lockdown_patterns'][group][activity][2]
		)



with open('../alphas/lockdown_%d.yaml'%lockdown, 'w') as file:
    yaml.dump(alphas, file)