# Covid Optimization

Implementation of Covid Testing Optimization

Before starting, you should install requirements:
```
pip install -r requirements.txt
pip install -e gym-covid

```
To run the dynamics symply go to the main folder and do

```
python run_dynamics.py --heuristic homogeneous --a_tests 0 --m_tests 0 --policy_params 444442
```

Things to note:
* Make sure you use Python3
* The ```a_tests``` and ```m_tests``` flags correspond to the number of antibody and molecular tests
* The number of days is set to 182 by default, you can change this manually in ```run_dynamics.py```
* The ```policy_params``` indicated the lockdown for each group. For example, if you add 010202, it means that age groups 1,3,5 are in lockdown pattern 0, group 2 is in lockdown 1 and groups 4 and 5 are in lockdown 2.
* The ```heuristic``` flag chooses the test heuristic, which can be equal to ```homogeneous```, ```age_group_i``` where i ranges from 1 to 6 and gives all testing to group i, ```forecasting_heuristic``` or ```random```
* The command produces a plot that is stored in the ```results_runs``` folder


# Lockdown simulation

To run the lockdown simulation do

```
python sim_lockdown.py --heuristic homogeneous --a_tests 0 --m_tests 0 --lockdown_start 90 --n_days 182 --initial_infected 1
```

* The ```heuristic```, ```a_tests``` and ```m_tests``` flags are the same as in the previous section
* The ```lockdown_start``` flag indicates the day that lockdown started
* The ```n_days``` flag indicates how many days to run the simulation
* The ```initial_infected``` flag is the number of people infected at day zero (we assume these people to be sampled from the groups in proportion with the size of the group)
* The script outputs a CSV file with the number of infected people, deaths, etc, at each day, the script also outputs a plot of the dynamics (you can find them in the folder ```results_runs```)
