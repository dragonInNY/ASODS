#!/bin/sh

#SBATCH --account=free    
#SBATCH --job-name=Patent_greedy
#SBATCH --mem-per-cpu=2G   

#SBATCH -N 4
#SBATCH --time=1:00:00       

#SBATCH --exclusive

 
source activate ASODS
mpirun -np 96 python /rigel/home/hy2726/codes/tests/greedy_test.py
# End of script