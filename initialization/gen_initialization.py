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

# Construct initialization
initialization_dict = {}
for group in data_dict['age_groups']:
      initialization_dict[group] = {
            "S": float(initial_percentages["S"]*data_dict["population"][group][region]),
            "E": float(initial_percentages["E"]*data_dict["population"][group][region]),
            "I": float(initial_percentages["I"]*data_dict["population"][group][region]),
            "R": float(initial_percentages["R"]*data_dict["population"][group][region]),
            "Ia": float(initial_percentages["Ia"]*data_dict["population"][group][region]),
            "Ips": float(initial_percentages["Ips"]*data_dict["population"][group][region]),
            "Ims": float(initial_percentages["Ims"]*data_dict["population"][group][region]),
            "Iss": float(initial_percentages["Iss"]*data_dict["population"][group][region]),
            "Rq": float(initial_percentages["Rq"]*data_dict["population"][group][region]),
            "H": float(initial_percentages["H"]*data_dict["population"][group][region]),
            "ICU": float(initial_percentages["ICU"]*data_dict["population"][group][region]),
            "D": float(initial_percentages["D"]*data_dict["population"][group][region]),
      }


with open('../initialization/initialization.yaml', 'w') as file:
    yaml.dump(initialization_dict, file)
