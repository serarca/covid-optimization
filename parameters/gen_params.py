import pickle
import pprint
import yaml
pp = pprint.PrettyPrinter(indent=4)
import argparse
# Parse data
parser = argparse.ArgumentParser()
parser.add_argument("-region", "--region", help="Region")
args = parser.parse_args()

# Import data
data_dict = pickle.load( open( "../data/data_dict.p", "rb" ) )


# Choose default parameters
region = args.region



# Assume all of the groups have the following initial distribution
initial_percentages = {
	  "S": 0.998,
      "E": 0.001,
      "I": 0.001,
      "R": 0,
      "Ia": 0,
      "Ips": 0,
      "Ims": 0,
      "Iss": 0,
      "Rq": 0,
      "H": 0,
      "ICU": 0,
      "D": 0,
}

# Create yaml
yaml_dict ={}
yaml_dict["seir-groups"] = {}
for group in data_dict['age_groups']:
	yaml_dict["seir-groups"][group] = {
		"name":group,
		"parameters":{
			"beta": float(data_dict['beta']),
			"sigma": float(data_dict['SEIR_parameters']['sigma'][group]),
			"mu": float(data_dict['SEIR_parameters']['mu'][group]),
			"p_H": float(data_dict['SEIR_parameters']['p_H'][group]),
			"p_ICU": float(data_dict['SEIR_parameters']['p_ICU'][group]),
			"p_Ia": float((1-data_dict['SEIR_parameters']['p_ss'][group])*0.5),
			"p_Ips": float((1-data_dict['SEIR_parameters']['p_ss'][group])*0.25),
			"p_Ims": float((1-data_dict['SEIR_parameters']['p_ss'][group])*0.25),
			"p_Iss": float(data_dict['SEIR_parameters']['p_ss'][group]),
			"lambda_H_R": float(data_dict['SEIR_parameters']['lambda_HR'][group]),
			"lambda_H_D": float(data_dict['SEIR_parameters']['lambda_HD'][group]),
			"lambda_ICU_R": float(data_dict['SEIR_parameters']['lambda_ICUR'][group]),
			"lambda_ICU_D": float(data_dict['SEIR_parameters']['lambda_ICUD'][group]),
		},
		"economics":{
			"work_value": float(data_dict['economic_value'][(region,group)]),
			"lockdown_fraction": float(data_dict['lockdown_fraction']),
			"death_value": float(data_dict['death_cost'][group]),
		},
		"contacts":{
			activity:{
				group2: float(data_dict['social_contact_matrices'][activity][group2][group]) for group2 in data_dict['age_groups']
			} for activity in data_dict['activities']
		}
	}

# Add ICUS, beds and test capacities
yaml_dict['global-parameters'] = {
	'C_H': float(data_dict['hospital_icu']["Hospital"][region]),
	'C_ICU':float(data_dict['hospital_icu']["ICU"][region]),
}

with open('../parameters/%s.yaml'%(region), 'w') as file:
    yaml.dump(yaml_dict, file)

