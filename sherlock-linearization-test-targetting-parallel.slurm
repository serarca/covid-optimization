#!/bin/bash

 

# Example of running a job array to run Gurobi optimization based on a config file

# Running from project space instead of home is recommended

 

#SBATCH --array=8,14,16,17,19,20,21,24,25,28,32,34,38,40,59,61,64,79,80,102,135,138,140,149,150,152,160,198 # there is a max array size - 512 tasks

#SBATCH -J test_target

#SBATCH -p normal




#SBATCH -t 0-15


#SBATCH -o parallel_runs_outputs_test_targetting/linearization_heur_test_targetting_runs-%A-%a.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=xwarnes@stanford.edu

 

#SBATCH -n 1

# SBATCH --mem-per-cpu=7gb


# Load software
ml load python/3.6.1
ml load viz
ml load py-matplotlib/3.2.1_py36
ml load py-pandas/1.0.3_py36
ml gurobi/9.0.3_py36
# export GRB_LICENSE_FILE=$HOME/gurobi.lic
ml load py-scipy/1.4.1_py36

# Add path to your miniconda and activate the conda env

# export PATH=$PROJECT_HOME/miniconda3/bin:$PATH

# activate conda env where all of your python packages are installed

# source activate myenv

srun --exclusive -n 1 python3 run_linearization_heuristic_test_targetting_value.py $SLURM_ARRAY_TASK_ID

# for i in {0..29}; do
#     srun --exclusive -n 1 python3 run_linearization_heuristic.py $((SLURM_ARRAY_TASK_ID * 50 + i)) &
# done

# wait # important to make sure the job doesn't exit before the background tasks are done


# Run python with a command line arg being an index from 1 to 300

# srun python3 test_linearization_heuristic_metaParam.py $SLURM_ARRAY_TASK_ID
