# Covid Optimization

Implementation of Covid Testing Optimization

Before starting, you should install requirements:
```
pip install -r requirements.txt

```

# Main Dynamical Model

The main dynamical model is in ```group.py```. To see how to use it, you can run ```example.py```, for example, which simulates a the aproximate policy of the French government, with homogenous testing.

```
python example.py
```

# Linearization Heuristic

You can also run the linearization heuristic by running ```run_linearizartion_heurisitic```. This runs the heuristic specified in ```linearization.py``` (see heuristics folder). This runs the heuristic and saves the result as a pickle object in the linearization_dyn_models folder. To plot the results, you can run from that folder, 

```
python plot.py -i file_name
```

where ```file_name``` is the name of the pickled dynamical model to be plotted. 