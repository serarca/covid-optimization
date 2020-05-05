# Testing

To run a test use ```test.py``` and use ```-data``` as the flag for the input data. Script will produce a graph of dynamics

```
python test.py -data case2.yaml
```

Case 1:
* Two populations with exactly the same parameters
* Young population is initially infected, old population is not
* Matrix of contacts is homogeneous (all entries are the same)
* No testing
* Unlimited beds and ICUs

Case 2:
* Same as Case 1, but limited beds to 20 and ICUs to 10

Case 3
* Same as Case 2, but each day, 10 molecular tests are applied to each group

Case 4
* Same as Case 2, but each day, 10 antibody tests are applied to each group

Case 4
* Same as Case 2, but each day, 10 antibody tests and 10 molecular tests are applied to each group

To run a test on rho use ```python test_rho.py -data case4.yaml```