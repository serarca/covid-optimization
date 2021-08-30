import yaml
from inspect import getsourcefile
import os.path
import os


runs = []
for n in os.listdir():
    if n != "check_errors.py" and n != "delete_errors.py" and n!= "check_infeasible.py" and n!= "check_runs.py":
        a = n.split("-")
        if len(a)<2:
            print(a)
        print(a[2][:-4])
        run_number = a[2][:-4]
        
        runs.append(int(run_number))
        # with open(n) as file:
        #     if "srun: error" in file.read():
        #         error_runs.append(int(run_number))

all_runs = set(range(19500))

print(len(sorted(list(all_runs.difference(set(runs))))))
print(sorted(list(all_runs.difference(set(runs)))))
    