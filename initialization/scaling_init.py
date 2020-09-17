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
old_init = yaml.load(open( "60days.yaml", "rb" ))
scaling = 1000.0

# Construct initialization
scaled_init_dict = {}
for group in old_init:
      scaled_init_dict[group] = {
            "S": old_init[group]["S"] / scaling,
            "E": old_init[group]["E"] / scaling,
            "I": old_init[group]["I"] / scaling,
            "R": old_init[group]["R"] / scaling,
            "Ia": old_init[group]["Ia"] / scaling,
            "Ips": old_init[group]["Ips"] / scaling,
            "Ims": old_init[group]["Ims"] / scaling,
            "Iss": old_init[group]["Iss"] / scaling,
            "Rq": old_init[group]["Rq"] / scaling,
            "H": old_init[group]["H"] / scaling,
            "ICU": old_init[group]["ICU"] / scaling,
            "D": old_init[group]["D"] / scaling,
      }

with open('../initialization/60days-scaled.yaml', 'w') as file:
    yaml.dump(scaled_init_dict, file)