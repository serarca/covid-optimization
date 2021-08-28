#!/bin/bash

 

# Example of running a job array to run Gurobi optimization based on a config file

# Running from project space instead of home is recommended


#SBATCH -J python_checker

#SBATCH -p normal

#SBATCH -t 2-0


#SBATCH -o python_scheduler/python_scheluder_output-%A.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=xwarnes@stanford.edu

 

#SBATCH -n 1

# SBATCH --mem-per-cpu=1GB


# Load software
ml load python/3.6.1

# Add path to your miniconda and activate the conda env

# export PATH=$PROJECT_HOME/miniconda3/bin:$PATH

# activate conda env where all of your python packages are installed

# source activate myenv

srun --exclusive -n 1 python3 run_scheduler_sherlock.py

# wait # important to make sure the job doesn't exit before the background tasks are done
