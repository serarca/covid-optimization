#!/bin/bash

 

# Example of running a job array to run Gurobi optimization based on a config file

# Running from project space instead of home is recommended

 

#SBATCH --array=200-250%18  # there is a max array size - 512 tasks

#SBATCH -J run_linearization_heuristic

#SBATCH -p normal




#SBATCH -t 2-00


#SBATCH -o parallel_runs_outputs/linearization_heur_runs-%A-%a.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=xwarnes@stanford.edu

 

#SBATCH -n 1

#SBATCH --mem-per-cpu=7gb


# Load software


module load gurobipy3
export GRB_LICENSE_FILE=/home/users/xwarnes/gurobi-lic/yen10/gurobi.lic


srun --exclusive -n 1 python3 run_linearization_heuristic.py $SLURM_ARRAY_TASK_ID


# for i in {0..29}; do
#     srun --exclusive -n 1 python3 run_linearization_heuristic.py $((SLURM_ARRAY_TASK_ID * 50 + i)) &
# done

wait # important to make sure the job doesn't exit before the background tasks are done


# Run python with a command line arg being an index from 1 to 300

# srun python3 test_linearization_heuristic_metaParam.py $SLURM_ARRAY_TASK_ID
