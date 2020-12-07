#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 French_trigger_benchmark_ref.py --delta 1.000000 --icus 3200 --eta 0.200000 --groups all --xi 3719903.000000 --a_tests 0 --m_tests 0
python3 ICU_admissions_trigger_benchmark_ref.py --delta 1.000000 --icus 3200 --eta 0.200000 --groups all --xi 3719903.000000 --a_tests 0 --m_tests 0
