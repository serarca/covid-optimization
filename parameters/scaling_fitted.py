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
old_fitted = yaml.load(open( "fitted.yaml", "rb" ))
scaling = 10000.0

scaled_fitted = dict(old_fitted)

# Scale global_param
scaled_fitted["global-parameters"]["C_H"] = scaled_fitted["global-parameters"]["C_H"] / scaling

scaled_fitted["global-parameters"]["C_ICU"] = scaled_fitted["global-parameters"]["C_ICU"] / scaling



for group_h in scaled_fitted["seir-groups"]:
    # # Scale contacts
    # for act in scaled_fitted["seir-groups"][group_h]["contacts"]:
    #     for group_g in scaled_fitted["seir-groups"][group_h]["contacts"][act]:
    #         scaled_fitted["seir-groups"][group_h]["contacts"][act][group_g] = scaled_fitted["seir-groups"][group_h]["contacts"][act][group_g] * scaling
    
    # Scale econ death value
    scaled_fitted["seir-groups"][group_h]["economics"]["death_value"] = scaled_fitted["seir-groups"][group_h]["economics"]["death_value"] * scaling
        

with open('fitted-scaled.yaml', 'w') as file:
    yaml.dump(scaled_fitted, file)