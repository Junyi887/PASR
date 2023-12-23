#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --account=dasrepo_g
#SBATCH --job-name=ns_lres_sim_s4_0
#SBATCH -C gpu&hbm80g
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=Joey000122@gmail.com
#SBATCH --mail-type=ALL

module load pytorch/2.0.1

cmd1="srun python train.py --data_path /pscratch/sd/j/junyi012/NS_lrsim_s4 --data ns_lres_sim_s4 --in_channels 3 --model PASR_ODE_small --lr 0.001 --batch_size 4 --epochs 500 --lr_step 100 --gamma 0.5 --seed 2407 --ode_method rk4 --lamb_p 0 --physics False --timescale_factor 4 --upsampler nearest_conv --n_snapshots 10"

set -x
bash -c "$cmd1"
