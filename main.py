import yaml
from group import SEIR_group

with open("parameters.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)


# Create groups from parameters
groups = {}
for g in parameters['seir-groups']:
	groups[g] = SEIR_group(parameters['seir-groups'][g])

# Attach other groups to each group
for g in groups:
	groups[g].attach_other_groups(groups)