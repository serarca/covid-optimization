#!/bin/bash

 

# Example of running a job array to run Gurobi optimization based on a config file

# Running from project space instead of home is recommended

 

#SBATCH --array=5,6,7,8             # there is a max array size - 512 tasks

#SBATCH -J linearization_heur_runs

#SBATCH -p normal

#SBATCH -n 1

#SBATCH -c 1                              # one core per task

#SBATCH -t 24:00:00

##SBATCH --mem=4gb

#SBATCH -o parallel_runs_outputs/linearization_heur_runs-%A-%a.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=xwarnes@stanford.edu

 

# Load software

module load gurobipy3



# Add path to your miniconda and activate the conda env

# export PATH=$PROJECT_HOME/miniconda3/bin:$PATH

# activate conda env where all of your python packages are installed

# source activate myenv

 

# Run python with a command line arg being an index from 1 to 300

srun python3 run_linearization_heuristic.py $SLURM_ARRAY_TASK_ID