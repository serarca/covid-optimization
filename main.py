import yaml
from group import SEIR_group, DynamicalModel

with open("parameters.yaml") as file:
    # The FullLoader parameter handles the conversion from YAML
    # scalar values to Python the dictionary format
    parameters = yaml.load(file, Loader=yaml.FullLoader)

# Set up parameters of simulation
dt = 1

DynamicalModel = DynamicalModel(parameters, dt)

DynamicalModel.upper_bound(2,[1,1],[1,1])