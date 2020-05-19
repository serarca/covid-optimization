# Covid Optimization

Implementation of Covid Testing Optimization

Before starting, you should install requirements:
```
pip install -r requirements.txt
pip install -e gym-covid
```
To run the dynamics symply go to the main folder and do

```
python run_dynamics.py --heuristic homogeneous --a_tests 0 --m_tests 0 --policy constant --policy_params 010202
```

Things to note:
* Make sure you use Python3 
* The ```a_tests``` and ```m_tests``` flags correspond to the number of antibody and molecular tests
* The number of days is set to 182 by default, you can change this manually in ```run_dynamics.py```
* The ```policy``` flag is the policy for lockdown, right now you can only use **constant** or **baseline**. If you use constant then you should add a string of numbers as the ```policy_params``` flag. For example, if you add 010202, it means that age groups 1,3,5 are in lockdown pattern 0, group 2 is in lockdown 1 and groups 4 and 5 are in lockdown 2. 
* If you use **baseline** as policy, this returns the best constant policy, which corresponds to the pattern 000120
* It is possible to use other policies, which come from RL algorithms, but for the moment no RL algorithm beats the baseline, so they require more work
* The ```heuristic``` flag chooses the test heuristic, which can be equal to ```homogeneous``` or to ```age_group_i``` where i ranges from 1 to 6
* The command produces a plot that is stored in the ```results_runs``` folder
