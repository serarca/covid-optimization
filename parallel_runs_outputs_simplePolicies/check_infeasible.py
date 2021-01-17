import yaml
from inspect import getsourcefile
import os.path
import os


error_runs = []
for n in os.listdir():
    if n != "check_errors.py" and n != "delete_errors.py" and n!= "check_infeasible.py":
        a = n.split("-")
        if len(a)<2:
            print(a)
        print(a[2][:-4])
        run_number = a[2][:-4]
        
        with open(n) as file:
            if "ERROR. Problem infeasible at time k=0. Halting..." in file.read():
                error_runs.append(int(run_number))

print(sorted(error_runs))
    