import argparse
import os

DATA_INFO = {
    'decay_turbulence':["/pscratch/sd/j/junyi012/Decay_Turbulence_small/","3"],
    "rbc":["/pscratch/sd/j/junyi012/RBC_small/","3"],
    "burgers2D":["/pscratch/sd/j/junyi012/burger2D_10/","3"],
    "decay_turbulence_coord":["/pscratch/sd/j/junyi012/Decay_Turbulence_small/","3"],
    "rbc_coord":["/pscratch/sd/j/junyi012/RBC_small/","3"],
    "burgers2D_coord":["/pscratch/sd/j/junyi012/burger2D_10/","3"],

}


def generate_bash_script(data_name, model, cmd_text=None):
    job_name = f"{data_name}_{model}"
    bash_content = f"""#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --account=dasrepo_g
#SBATCH --job-name={job_name}
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
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
    data_names = ["decay_turbulence_coord", "rbc_coord", "burgers2D_coord"]
    for data_name in data_names:
        data_path, channel = DATA_INFO[data_name]
        cmd_text = f"python baseline_FNO_v2.py --data_path {data_path} --in_channels {channel} --data {data_name} --epochs 500 --lr 0.001 --lr_step 100 --gamma 0.5 --batch_size 16 --seed 3407"
        jobname = generate_bash_script(data_name, "FNO", cmd_text=cmd_text)
        with open(f"run_FNO_v2.sh", "a") as f:
            f.write(f"sbatch bash_script/{jobname}.sbatch\n")