#!/bin/bash
#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J KT
#SBATCH --mail-user=joey000122@gmail.com
#SBATCH --mail-type=All
#SBATCH -t 23:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo $1

module load python
module load pytorch

srun -n 1 -c 256 --cpu_bind=cores python data_gen/data_gen_DT_abalation.py --res 1024 --seed 0
