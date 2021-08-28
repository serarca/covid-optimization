import time
import subprocess


def increase_job_array(l, job_upper_limit, number_of_parallel_jobs):
    lower_value = int(l.split('=')[1].split('-')[0])
    upper_value = int(l.split('=')[1].split('-')[1])

    if lower_value + 1 < job_upper_limit:
        new_lower_value = upper_value + 1
        new_upper_value = min(upper_value + number_of_parallel_jobs, job_upper_limit)
    
    return f"#SBATCH --array={new_lower_value}-{new_upper_value} \n"


def cycle_next_jobs(filename, job_upper_limit, number_of_parallel_jobs):
    old_file = open(filename, 'r')
    list_of_lines = old_file.readlines()

    print("read file")

    for i  in range(len(list_of_lines)):
        if '#SBATCH --array' in list_of_lines[i]:
            print(f"l includes the text l is:{list_of_lines[i]}")
            list_of_lines[i] = increase_job_array(list_of_lines[i], job_upper_limit, number_of_parallel_jobs)
            print(f"after increase l is: {list_of_lines[i]}")
    
    print(f"list of lines is: {list_of_lines}")
    
    new_file = open(filename, 'w')
    new_file.writelines(list_of_lines)
    new_file.close()




filename = 'sherlock-linearization-multiple-initial-conditions-parallel.slurm'

job_upper_limit = 19500
number_of_parallel_jobs = 499

while True:
    # Check if there are any processes running
    running_processes = subprocess.run(['squeue','-uxwarnes'], stdout=subprocess.PIPE).stdout.decode('utf-8')

    if 'xwarnes' not in running_processes:

        print("in inner if")

        cycle_next_jobs(filename, job_upper_limit, number_of_parallel_jobs)
        subprocess.run(['sbatch', filename])

        time.sleep(60)

        cycle_next_jobs(filename, job_upper_limit, number_of_parallel_jobs)
        subprocess.run(['sbatch', filename])


        

    time.sleep(3600)
