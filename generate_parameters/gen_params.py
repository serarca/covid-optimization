import pickle
import pprint
import yaml
pp = pprint.PrettyPrinter(indent=4)
import argparse
# Parse data
parser = argparse.ArgumentParser()
parser.add_argument("-lockdown", "--lockdown", help="Lockdown pattern")
parser.add_argument("-region", "--region", help="Region")
args = parser.parse_args()

# Import data
data_dict = pickle.load( open( "data_dict.p", "rb" ) )


# Choose default parameters
region = args.region
tests_A = 3000
tests_M = 3000



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
# Percentages in each lockdown pattern
with open("../lockdown_patterns/pattern_"+args.lockdown+".yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    patterns_percentages = yaml.load(file, Loader=yaml.FullLoader)


# Create yaml
yaml_dict ={}
yaml_dict["seir-groups"] = {}
for group in data_dict['age_groups']:
	for p in patterns_percentages[group]: 
		yaml_dict["seir-groups"][group+"_lp_"+str(p)] = {
			"name":group+"_lp_"+str(p),
			"initial-conditions":{
				"S": float(patterns_percentages[group][p]*initial_percentages["S"]*data_dict["population"][group][region]),
				"E": float(patterns_percentages[group][p]*initial_percentages["E"]*data_dict["population"][group][region]),
				"I": float(patterns_percentages[group][p]*initial_percentages["I"]*data_dict["population"][group][region]),
				"R": float(patterns_percentages[group][p]*initial_percentages["R"]*data_dict["population"][group][region]),
				"Ia": float(patterns_percentages[group][p]*initial_percentages["Ia"]*data_dict["population"][group][region]),
				"Ips": float(patterns_percentages[group][p]*initial_percentages["Ips"]*data_dict["population"][group][region]),
				"Ims": float(patterns_percentages[group][p]*initial_percentages["Ims"]*data_dict["population"][group][region]),
				"Iss": float(patterns_percentages[group][p]*initial_percentages["Iss"]*data_dict["population"][group][region]),
				"Rq": float(patterns_percentages[group][p]*initial_percentages["Rq"]*data_dict["population"][group][region]),
				"H": float(patterns_percentages[group][p]*initial_percentages["H"]*data_dict["population"][group][region]),
				"ICU": float(patterns_percentages[group][p]*initial_percentages["ICU"]*data_dict["population"][group][region]),
				"D": float(patterns_percentages[group][p]*initial_percentages["D"]*data_dict["population"][group][region]),
			},
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
			      "v_unconf": float(data_dict['economic_values'][(region, group, p)]),
			      "v_conf": float(data_dict['economic_values'][(region, group, p)]),
			      "v_deaths": float(data_dict['death_cost']['death_cost'][group]),
			},
			"contacts":{
				group2+"_lp_"+str(p2): float(data_dict["all_contacts"][(region,group,p,region,group2,p2)]) for group2 in data_dict['age_groups'] for p2 in patterns_percentages[group2]
			}
		}

# Add ICUS, beds and test capacities
yaml_dict['global-parameters'] = {
	'C_H': float(data_dict['hospital_icu']["Hospital"][region]),
	'C_ICU':float(data_dict['hospital_icu']["ICU"][region]),
	'A_tests':float(data_dict['testing'][region]['antibodies']),
	'M_tests':float(data_dict['testing'][region]['molecular']),
}


with open('../parameters/%s_lp_%s_params.yaml'%(region,args.lockdown), 'w') as file:
    documents = yaml.dump(yaml_dict, file)

