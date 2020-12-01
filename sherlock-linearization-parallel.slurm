#!/bin/bash

 

# Example of running a job array to run Gurobi optimization based on a config file

# Running from project space instead of home is recommended

 

#SBATCH --array=7,8,9,24,25,26,28,29,40,95,96,97,99,100,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,123,124,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,175,201,202,203,204,206,207,238,240,241,242,243,244,245,248,249,250,251,252,253,254,256,257,259,261,297,298,299,300,301,302,303,305,306,307,308,319,338,341,342,344,346,347,348,349,350,351,352,353,354,356,357,358,359,360,361,362,363,364,365,366,367,368,369,387,388,389,390,391,392,393,394,395,396,397,398,401,403,414,416,417,418,419,421,423,424,425,426,427,428,429,430,431,432,433,435,436,437,438,439,441,443,444,445,448,450,451 # there is a max array size - 512 tasks

#SBATCH -J run_linearization_heuristic

#SBATCH -p normal




#SBATCH -t 0-10


#SBATCH -o parallel_runs_outputs/linearization_heur_runs-%A-%a.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=xwarnes@stanford.edu

 

#SBATCH -n 1

#SBATCH --mem-per-cpu=7gb


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

srun --exclusive -n 1 python3 run_linearization_heuristic.py $SLURM_ARRAY_TASK_ID

# for i in {0..29}; do
#     srun --exclusive -n 1 python3 run_linearization_heuristic.py $((SLURM_ARRAY_TASK_ID * 50 + i)) &
# done

# wait # important to make sure the job doesn't exit before the background tasks are done


# Run python with a command line arg being an index from 1 to 300

# srun python3 test_linearization_heuristic_metaParam.py $SLURM_ARRAY_TASK_ID
