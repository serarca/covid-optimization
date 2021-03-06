#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

python3 ProcessMultipleFittings.py --identifier 1
