# Lower Bounds - Heuristics

We use heuristics to calculate a lower bound, the file ```simple_heuristics.py``` contains different simple heuristics. The file ```linearization.py``` contains the linearization heuristic. To run the linearization heuristic, go to the main folder and run 

```
python run_linearization_heuristic.py
```

to run the simple heuristics, you have to create a dynamical model (see ```group.py```) import ```simple_heuristics.py``` and run one of the simple hueristics found inside.

Current heuristics:
* all_to_one: Chooses a group randomly and allocates all testing to it
* random_distribution: Distributes the total amount of tests randomly on each unit of time
* homogenous: Distributes all testing homogenously among all groups.
* no_tests: Does not test.
* Linearization Heuristic: Decides both on lockdowns and testing based on linearizing the system dynamics around a nominal trajectory at each point in time and solving iteratively an LP formulation of the problem until the solutions converge (or a number of iterations is reached).