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
old_econ = yaml.load(open( "econ.yaml", "rb" ))
scaling = 1000.0
money_scaling = 10000.0

scaled_econ = dict(old_econ)

# Scale Econ cost of death
for group in scaled_econ["econ_cost_death"]:
      scaled_econ["econ_cost_death"][group] = (scaled_econ["econ_cost_death"][group] * scaling / money_scaling)

# Scale employment param

for group in scaled_econ["employment_params"]["v"]:
      scaled_econ["employment_params"]["v"][group]["leisure"] = scaled_econ["employment_params"]["v"][group]["leisure"] * scaling / money_scaling
      scaled_econ["employment_params"]["v"][group]["other"] = scaled_econ["employment_params"]["v"][group]["other"] * scaling / money_scaling
      scaled_econ["employment_params"]["v"][group]["transport"] = scaled_econ["employment_params"]["v"][group]["transport"] * scaling / money_scaling

# Scale schooling params

for group in scaled_econ["schooling_params"]:
      scaled_econ["schooling_params"][group] = scaled_econ["schooling_params"][group] * scaling / money_scaling


with open('econ-scaled.yaml', 'w') as file:
    yaml.dump(scaled_econ, file)