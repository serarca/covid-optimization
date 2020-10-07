#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

ml python/3.6.1
python3 simple_benchmarks.py --delta 0.500000 --icus 3200 --eta 0.000000 --groups all --xi 371990.300000 --a_tests 120000 --m_tests 120000
python3 constant_gradient_benchmarks.py --delta 0.500000 --icus 3200 --eta 0.000000 --groups all --xi 371990.300000 --a_tests 120000 --m_tests 120000
python3 dynamic_gradient_benchmarks.py --delta 0.500000 --icus 3200 --eta 0.000000 --groups all --xi 371990.300000 --a_tests 120000 --m_tests 120000
