import yaml
from inspect import getsourcefile
import os.path
import os



for n in os.listdir():
    if n != "check_errors.py" and n != "delete_errors.py" and n!= "check_infeasible.py":
        
        with open(n) as file:
            if "srun: error" in file.read():
                os.remove(n)
