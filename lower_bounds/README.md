# Lower Bounds

We use heuristics to calculate a lower bound, the file ```heuristics.py``` contains different heuristics. To run a heuristic use the command

```
python lower_bound.py --data test5.yaml --heuristic heuristic_name
```

Current heuristics:
* all_to_one: Chooses a group randomly and allocates all testing to it
* random_distribution: Distributes the total amount of tests randomly on each unit of time