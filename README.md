# Covid Optimization

Implementation of Covid Testing Optimization

Before starting, you should install requirements:
```
pip install -r requirements.txt
pip install -e gym-covid

```

# Main Dynamical Model

The main dynamical model is in ```group.py```. To see how to use it, you can run ```sim_lockdown```, for example, which simulates a system where the economy is open, and on a specific day, the government moves it into lockdown. To run this simulation do

```
python sim_lockdown.py --heuristic homogeneous --a_tests 0 --m_tests 0 --lockdown_start 90 --n_days 182 --initial_infected 1
```

* The ```heuristic```, ```a_tests``` and ```m_tests``` flags are the same as in the previous section
* The ```lockdown_start``` flag indicates the day that lockdown started
* The ```n_days``` flag indicates how many days to run the simulation
* The ```initial_infected``` flag is the number of people infected at day zero (we assume these people to be sampled from the groups in proportion with the size of the group)
* The script outputs a CSV file with the number of infected people, deaths, etc, at each day, the script also outputs a plot of the dynamics (you can find them in the folder ```results_runs```)

