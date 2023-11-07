import argparse
import os

DATA_INFO = {
    'burger_2D':["../burger2D_10",3],
    'decay_turb':["/pscratch/sd/j/junyi012/Decay_Turbulence_small",3],
    "rbc":["RBC_small",3],
}


def generate_bash_script(data_name, model,scale_factor, cmd_text):
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
    data_names = ["s2_sig2", "s2_sig4", "s2_sig1", "s2_sig0"]
    methods = ["rk4", "euler",""]
    models = ["train_climate_ODE_wrapper.py", "baseline_conv_LSTM_Climate.py", "baseline_FNO_climate.py"]
    models2 = ["NODE", "LSTM", "FNO"]
    for data_name in data_names:
        data_path, scale_factor = DATA_INFO[data_name]
        for model,name in zip(models,models2):
            if model == "train_climate_ODE_wrapper.py":
                for method in methods:
                    cmd_text = f"python {model} --data_path {data_path} --batch_size 16 --epochs 400 --lr 0.001 --ode_method {method}"
                    job_name = generate_bash_script(data_name,str(name+method), scale_factor, cmd_text)
                    with open(f"run_climate.sh", "a") as f:
                        f.write(f"sbatch bash_script/{job_name}.sbatch\n")
            else:
                cmd_text = f"python {model} --data_path {data_path}"
                job_name = generate_bash_script(data_name, name,scale_factor, cmd_text)
                with open(f"run_climate.sh", "a") as f:
                    f.write(f"sbatch bash_script/{job_name}.sbatch\n")
