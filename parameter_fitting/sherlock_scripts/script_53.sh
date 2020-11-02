#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 ParameterFittingRandomToPython.py --days 60 --alpha 2.300000
