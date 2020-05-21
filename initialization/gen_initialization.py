import pickle
import pprint
import yaml
pp = pprint.PrettyPrinter(indent=4)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-region", "--region", help="Region")
args = parser.parse_args()

region = args.region

# Import data
data_dict = pickle.load( open( "../data/data_dict.p", "rb" ) )


# Construct initialization
initialization_dict = {}
for group in data_dict['age_groups']:
      initialization_dict[group] = {
            "S": float(data_dict["population"][group][region]),
            "E": float(0),
            "I": float(0),
            "R": float(0),
            "Ia": float(0),
            "Ips": float(0),
            "Ims": float(0),
            "Iss": float(0),
            "Rq": float(0),
            "H": float(0),
            "ICU": float(0),
            "D": float(0),
      }

with open('../initialization/patient_zero.yaml', 'w') as file:
    yaml.dump(initialization_dict, file)