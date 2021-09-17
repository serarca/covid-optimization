#!/bin/bash

 

# Example of running a job array to run Gurobi optimization based on a config file

# Running from project space instead of home is recommended

 
# there is a max array size - 512 tasks
#SBATCH --array=19325,19326,19327,19328,19329,19330,19331,19332,19333,19334,19335,19336,19337,19338,19339,19340,19341,19342,19343,19344,19345,19346,19347,19348,19349,19350,19351,19352,19353,19354,19355,19356,19357,19358,19359,19360,19361,19362,19363,19364,19365,19366,19367,19368,19369,19370,19371,19372,19373,19374,19375,19376,19377,19378,19379,19380,19381,19382,19383,19384,19385,19386,19387,19388,19389,19390,19391,19392,19393,19394,19395,19396,19397,19398,19399,19400,19401,19402,19403,19404,19405,19406,19407,19408,19409,19410,19411,19412,19413,19414,19415,19416,19417,19418,19419,19420,19421,19422,19423,19424,19425,19426,19427,19428,19429,19430,19431,19432,19433,19434,19435,19436,19437,19438,19439,19440,19441,19442,19443,19444,19445,19446,19447,19448,19449,19450,19451,19452,19453,19454,19455,19456,19457,19458,19459,19460,19461,19462,19463,19464,19465,19466,19467,19468,19469,19470,19471,19472,19473,19474,19475,19476,19477,19478,19479,19480,19481,19482,19483,19484,19485,19486,19487,19488,19489,19490,19491,19492,19493,19494,19495,19496,19497,19498,19499

#SBATCH -J run_mult_init_cond_linearization_heuristic

#SBATCH -p normal




#SBATCH -t 0-20


#SBATCH -o parallel_runs_mult_init_cond_outputs/linearization_heur_mult_init_cond_runs-%A-%a.out

#SBATCH --mail-type=ALL

#SBATCH --mail-user=xwarnes@stanford.edu

 

#SBATCH -n 1

# SBATCH --mem-per-cpu=7GB


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

srun --exclusive -n 1 python3 run_linearization_heuristic_multiple_init_cond.py $SLURM_ARRAY_TASK_ID

# for i in {0..29}; do
#     srun --exclusive -n 1 python3 run_linearization_heuristic.py $((SLURM_ARRAY_TASK_ID * 50 + i)) &
# done

# wait # important to make sure the job doesn't exit before the background tasks are done


# Run python with a command line arg being an index from 1 to 300

# srun python3 test_linearization_heuristic_metaParam.py $SLURM_ARRAY_TASK_ID
