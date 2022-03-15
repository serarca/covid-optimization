#!/bin/bash

 

# Example of running a job array to run Gurobi optimization based on a config file

# Running from project space instead of home is recommended

 
# there is a max array size - 512 tasks
#SBATCH --array=22306,22307,22308,22309,22310,22311,22312,22313,22314,22315,22316,22317,22318,22319,22320,22321,22322,22323,22324,22325,22326,22327,22328,22329,22330,22331,22332,22333,22334,22335,22336,22337,22338,22339,22340,22341,22342,22343,22344,22345,22346,22347,22348,22349,22350,22351,22352,22353,22354,22355,22356,22357,22358,22359,22360,22361,22362,22363,22364,22365,22366,22367,22368,22369,22370,22371,22372,22373,22374,22375,22376,22377,22378,22379,22380,22381,22382,22383,22384,22385,22386,22387,22388,22389,22390,22391,22392,22393,22394,22395,22396,22397,22398,22399,22400,22401,22402,22403,22404,22405,22406,22407,22408,22409,22410,22411,22412,22413,22414,22415,22416,22417,22418,22419,22420,22421,22422,22423,22424,22425,22426,22427,22428,22429,22430,22431,22432,22433,22434,22435,22436,22437,22438,22439,22440,22441,22442,22443,22444,22445,22446,22447,22448,22449,22450,22451,22452,22453,22454,22455,22456,22457,22458,22459,22460,22461,22462,22463,22464,22465,22466,22467,22468,22469,22470,22471,22472,22473,22474,22475,22476,22477,22478,22479,22480,22481,22482,22483,22484,22485,22486,22487,22488,22489,22490,22491,22492,22493,22494,22495,22496,22497,22498,22499

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
