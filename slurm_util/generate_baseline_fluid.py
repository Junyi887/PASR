import argparse
import os

DATA_INFO = {
    'decay_turbulence':["/pscratch/sd/j/junyi012/Decay_Turbulence_small/","3"],
    "rbc":["/pscratch/sd/j/junyi012/RBC_small/","3"],
    "burgers2D":["/pscratch/sd/j/junyi012/burger2D_10/","3"],
    "decay_turbulence_coord":["/pscratch/sd/j/junyi012/Decay_Turbulence_small/","3"],
    "DT_lrsim_256_s4_v0":["/pscratch/sd/j/junyi012/DT_lrsim_256_s4_v0","3","32"],
    "DT_lrsim_512_s4_v0":["/pscratch/sd/j/junyi012/DT_lrsim_512_s4_v0","3","32"],
    "DT_lrsim_1024_s4_v0":["/pscratch/sd/j/junyi012/DT_lrsim_1024_s4_v0","3","4"],
}


def generate_bash_script(data_name, model, cmd_text=None):
    job_name = f"{data_name}_{model}"
    bash_content = f"""#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --account=dasrepo_g
#SBATCH --job-name={job_name}
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=Joey000122@gmail.com
#SBATCH --mail-type=ALL

module load pytorch/2.0.1

set -x
srun {cmd_text}

"""

    os.makedirs("bash_script", exist_ok=True)  # Ensure directory exists
    with open(f"bash_script/{job_name}.sbatch", 'w') as out_file:
        out_file.write(bash_content)
        print(f"Bash script generated as {job_name}.sbatch")

    return job_name


# Run the function
if __name__ == "__main__":
    raw_names = ["DT_lrsim_256_s4_v0","DT_lrsim_512_s4_v0","DT_lrsim_1024_s4_v0"]
    data_names = ["DT_lrsim_256_s4_v0_FNO","DT_lrsim_512_s4_v0_FNO","DT_lrsim_1024_s4_v0_FNO"]
    for i,(raw_name,data_name) in enumerate (zip(raw_names,data_names)):
        data_path, channel,batch = DATA_INFO[raw_name]
        for ic in [3]:
            cmd_text = f"python baseline_FNO.py --data_path {data_path} --in_channels {ic} --data {data_name} --epochs 500 --lr 0.001 --lr_step 100 --gamma 0.5 --batch_size 32 --seed 3407 --n_snapshots 20 --timescale_factor 10"
            jobname = generate_bash_script(data_name, "FNO", cmd_text=cmd_text)
            with open(f"run_FNO.sh", "a") as f:
                f.write(f"sbatch bash_script/{jobname}.sbatch\n")
    data_names2 = ["DT_lrsim_256_s4_v0_ConvLSTM","DT_lrsim_512_s4_v0_ConvLSTM","DT_lrsim_1024_s4_v0_ConvLSTM"]
    for i,(raw_name,data_name) in enumerate (zip(raw_names,data_names2)):
        data_path, channel,batch = DATA_INFO[raw_name]
        for ic in [3]:
            cmd_text = f"python baseline_ConvLSTM.py --data_path {data_path} --in_channels {ic} --data {data_name} --batch_size {batch} --seed 3407 --n_snapshots 20 --timescale_factor 10"
            jobname = generate_bash_script(data_name, "FNO", cmd_text=cmd_text)
            with open(f"run_ConvLSTM.sh", "a") as f:
                f.write(f"sbatch bash_script/{jobname}.sbatch\n")