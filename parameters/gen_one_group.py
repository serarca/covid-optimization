import yaml
from copy import deepcopy


# Read group parameters
with open("../parameters/fitted.yaml") as file:
    universe_params = yaml.load(file, Loader=yaml.FullLoader)

# Read initialization
with open("../initialization/60days.yaml") as file:
	initialization = yaml.load(file, Loader=yaml.FullLoader)
	start_day = 60

# Read econ parameters
with open("../parameters/econ.yaml") as file:
	econ_params = yaml.load(file, Loader=yaml.FullLoader)


print(universe_params.keys())

all_activities = ["home","leisure","other","school","transport","work"]
econ_activities = ["leisure","other","transport"]
age_groups = [
	"age_group_0_9",
	"age_group_10_19",
	"age_group_20_29",
	"age_group_30_39",
	"age_group_40_49",
	"age_group_50_59",
	"age_group_60_69",
	"age_group_70_79",
	"age_group_80_plus",
]

# Calculate proportions of groups
proportions = {}
total_n = 0
for ag in initialization:
	proportions[ag] = initialization[ag]["N"]
	total_n += proportions[ag]
for ag in initialization:
	proportions[ag] = proportions[ag]/total_n
print(proportions)

# Duplicate
one_group_universe = deepcopy(universe_params)
del one_group_universe["seir-groups"]

one_group_universe['seir-groups'] = {"all_age_groups":{}}


# Fill up contacts
partial_contacts = {}
for ag in universe_params['seir-groups']:
	partial_contacts[ag] = {}
	for act in all_activities:
		partial_contacts[ag][act] = sum([universe_params['seir-groups'][ag]["contacts"][act][ag2] for ag2 in age_groups])

one_group_universe['seir-groups']["all_age_groups"]["contacts"] = {}
for act in all_activities:
	one_group_universe['seir-groups']["all_age_groups"]["contacts"][act] = {}
	one_group_universe['seir-groups']["all_age_groups"]["contacts"][act]["all_age_groups"] = sum([partial_contacts[ag][act] * proportions[ag] for ag in age_groups])

# Fill other info
one_group_universe['seir-groups']["all_age_groups"]["name"] = "all_age_groups"

# Calculate parameters
one_group_universe['seir-groups']["all_age_groups"]['parameters'] = {}
for par in universe_params['seir-groups']["age_group_0_9"]['parameters']:
	one_group_universe['seir-groups']["all_age_groups"]['parameters'][par] = sum([universe_params['seir-groups'][ag]['parameters'][par]*proportions[ag] for ag in age_groups])

with open('../parameters/one_group_fitted.yaml', 'w') as file:
    yaml.dump(one_group_universe, file)


# Now fill up the econ parameters of just one group
one_group_econ = deepcopy(econ_params)

one_group_econ["econ_cost_death"] = {"all_age_groups":sum([econ_params["econ_cost_death"][ag]*proportions[ag] for ag in age_groups])}
one_group_econ["schooling_params"] = {"all_age_groups":sum([econ_params["schooling_params"][ag]*proportions[ag] for ag in age_groups])}


one_group_econ["employment_params"]["v"] = {"all_age_groups":{}}
for act in econ_activities:
	one_group_econ["employment_params"]["v"]["all_age_groups"][act] = sum([econ_params["employment_params"]["v"][ag][act]*proportions[ag] for ag in age_groups])

with open('../parameters/one_group_econ.yaml', 'w') as file:
    yaml.dump(one_group_econ, file)

# Now do the different initializations
init_files = ["10days","20days","30days","40days","50days","60days","70days","80days","90days","100days"]
for file_name in init_files:
	with open("../initialization/%s.yaml"%file_name) as file:
		initialization = yaml.load(file, Loader=yaml.FullLoader)
	one_group_init = {"all_age_groups":{}}
	for comp in initialization["age_group_0_9"]:
		one_group_init["all_age_groups"][comp] = sum([initialization[ag][comp] for ag in age_groups])
	with open("../initialization/%s_one_group.yaml"%file_name, 'w') as file:
	    yaml.dump(one_group_init, file)






