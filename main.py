import yaml
from group import SEIR_group

with open("parameters.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)

# Set up parameters of simulation
dt = 1

# Create groups from parameters
groups = {}
for g in parameters['seir-groups']:
	groups[g] = SEIR_group(parameters['seir-groups'][g], dt)

# Attach other groups to each group
for g in groups:
	groups[g].attach_other_groups(groups)


# Testing
for g in groups:
	groups[g].take_time_step(1,1)