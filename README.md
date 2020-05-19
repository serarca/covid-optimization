# Covid Optimization

Implementation of Covid Testing Optimization

Before starting, should install requirements:
```
pip install -r requirements.txt
pip install -e gym-covid
```
To run the dynamics symply go to the main folder and do

```
python run_dynamics.py --lockdown 3 --region Ile-de-France --a_tests 0 --m_tests 0 --heuristic homogeneous
```
Make sure you use Python2. The ```a_tests``` and ```m_tests``` flags correspond to the number of antibody and molecular tests, the ```days``` flag corresponds to the number of days to run the simulation, the ```lockdown``` flag indicates the lockdown pattern, lockdown patterns are found in the ```lockdown_patterns``` folder. Finally, the ```heuristic``` flag chooses the test heuristic, which can be equal to ```homogeneous``` or to ```age_group_i``` where i ranges from 1 to 6. The command produces a plot that is stored in the ```results_runs``` folder
