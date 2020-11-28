#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 simple_benchmarks.py --delta 0.500000 --icus 2900 --eta 0.000000 --groups all --xi 5579854.500000 --a_tests 60000 --m_tests 60000
python3 time_gradient_benchmarks.py --delta 0.500000 --icus 2900 --eta 0.000000 --groups all --xi 5579854.500000 --a_tests 60000 --m_tests 60000
python3 age_group_gradient_benchmarks.py --delta 0.500000 --icus 2900 --eta 0.000000 --groups all --xi 5579854.500000 --a_tests 60000 --m_tests 60000
python3 activity_gradient_benchmarks.py --delta 0.500000 --icus 2900 --eta 0.000000 --groups all --xi 5579854.500000 --a_tests 60000 --m_tests 60000
python3 dynamic_gradient_benchmarks.py --delta 0.500000 --icus 2900 --eta 0.000000 --groups all --xi 5579854.500000 --a_tests 60000 --m_tests 60000
