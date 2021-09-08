import time
import subprocess
import os
import os.path
import sys


def increase_job_array(l, job_upper_limit, number_of_parallel_jobs, list_of_unfinished_jobs):
    
    # lower_value = int(l.split('=')[1].split('-')[0])
    # upper_value = int(l.split('=')[1].split('-')[1])

    # if lower_value + 1 < job_upper_limit:
    #     new_lower_value = upper_value + 1
    #     new_upper_value = min(upper_value + number_of_parallel_jobs, job_upper_limit)
    
    list_of_jobs_to_run = ""
    for i in range(min(number_of_parallel_jobs, len(list_of_unfinished_jobs))):
        list_of_jobs_to_run += str(list_of_unfinished_jobs[i])+","

    list_of_jobs_to_run = list_of_jobs_to_run[:-1]

    print(list_of_jobs_to_run)

    return f"#SBATCH --array={list_of_jobs_to_run}\n"

def calculate_missing_jobs(foldername):
    runs = []
    for n in os.listdir(foldername):
        if n != "check_errors.py" and n != "delete_errors.py" and n!= "check_infeasible.py" and n!= "check_runs.py":
            a = n.split("-")
            # if len(a)<2:
                # print(a)
            # print(a[2][:-4])
            run_number = a[2][:-4]
            
            runs.append(int(run_number))
            # with open(n) as file:
            #     if "srun: error" in file.read():
            #         error_runs.append(int(run_number))

    all_runs = set(range(19500))

    if not sorted(list(all_runs.difference(set(runs)))):
        sys.exit("All jobs have been runned!!")

    print(len(sorted(list(all_runs.difference(set(runs))))))
    return sorted(list(all_runs.difference(set(runs))))



def cycle_next_jobs(filename, job_upper_limit, number_of_parallel_jobs, foldername):
    old_file = open(filename, 'r')
    list_of_lines = old_file.readlines()

    list_of_unfinished_jobs = calculate_missing_jobs(foldername)

    print("read file")

    for i  in range(len(list_of_lines)):
        if '#SBATCH --array' in list_of_lines[i]:
            print(f"l includes the text l is:{list_of_lines[i]}")
            list_of_lines[i] = increase_job_array(list_of_lines[i], job_upper_limit, number_of_parallel_jobs, list_of_unfinished_jobs)
            print(f"after increase l is: {list_of_lines[i]}")
    
    print(f"list of lines is: {list_of_lines}")
    
    new_file = open(filename, 'w')
    new_file.writelines(list_of_lines)
    new_file.close()




filename = 'sherlock-linearization-multiple-initial-conditions-parallel.slurm'
foldername = 'parallel_runs_mult_init_cond_outputs'
job_upper_limit = 19500
max_number_of_parallel_jobs = 499

while True:
    # Check if there are any processes running
    all_running_processes = subprocess.run(["squeue", "-uxwarnes"], stdout=subprocess.PIPE).stdout.decode('utf-8')

    print(all_running_processes)

    # break

    number_of_PD_processes = sum([1  for l in all_running_processes.split() if l=='PD'])

    number_of_CF_processes = sum([1  for l in all_running_processes.split() if l=='CF'])

    number_of_running_processes = sum([1  for l in all_running_processes.split() if l=='R'])


    print(f"number of processes = {number_of_running_processes}")

    if number_of_PD_processes == 0 and number_of_CF_processes == 0 and number_of_running_processes < 100:
        number_of_parallel_jobs = 500

        print("in inner if")
        print(f"running {number_of_parallel_jobs} parallel jobs")

        cycle_next_jobs(filename, job_upper_limit, number_of_parallel_jobs, foldername)
        subprocess.run(['sbatch', filename])

    time.sleep(1800)
