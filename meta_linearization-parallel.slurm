#!/bin/bash

 

# Example of running a job array to run Gurobi optimization based on a config file

# Running from project space instead of home is recommended

 

#SBATCH --array=299   # there is a max array size - 512 tasks

#SBATCH -J meta_param_linearization_heur_runs

#SBATCH -p normal




#SBATCH -t 10:00:00


#SBATCH -o parallel_runs_outputs/meta_param_linearization_heur_runs-%A-%a.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=xwarnes@stanford.edu

 

#SBATCH -n 24

#SBATCH --mem-per-cpu=7gb


# Load software

module load gurobipy3



# Add path to your miniconda and activate the conda env

# export PATH=$PROJECT_HOME/miniconda3/bin:$PATH

# activate conda env where all of your python packages are installed

# source activate myenv

 
for i in {0..23}; do
    srun --exclusive -n 1 python3 test_linearization_heuristic_metaParam.py $((SLURM_ARRAY_TASK_ID+i)) &
done

wait # important to make sure the job doesn't exit before the background tasks are done


# Run python with a command line arg being an index from 1 to 300

# srun python3 test_linearization_heuristic_metaParam.py $SLURM_ARRAY_TASK_ID


