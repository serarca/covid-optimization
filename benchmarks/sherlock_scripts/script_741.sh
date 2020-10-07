#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 simple_benchmarks.py --delta 5.000000 --icus 2300 --eta 0.000000 --groups all --xi 0.000000 --a_tests 120000 --m_tests 120000
python3 constant_gradient_benchmarks.py --delta 5.000000 --icus 2300 --eta 0.000000 --groups all --xi 0.000000 --a_tests 120000 --m_tests 120000
python3 dynamic_gradient_benchmarks.py --delta 5.000000 --icus 2300 --eta 0.000000 --groups all --xi 0.000000 --a_tests 120000 --m_tests 120000
