import argparse


DATA_INFO = {'decay_turbulence':["/pscratch/sd/j/junyi012/Decay_Turbulence_small/","3"],
             "rbc":["/pscratch/sd/j/junyi012/RBC_small/","3"],
             "burgers2D":["/pscratch/sd/j/junyi012/burger2D_10/","3"],
            "decay_turbulence_lrsim":["../decay_turb_lrsim_short4","3"],
            "decay_turbulence_lrsim_v2":["../DT_shorter","3"],
            "era5":["/pscratch/sd/j/junyi012/climate_sigma4_v2",1],
            "era5_sequence":["/pscratch/sd/j/junyi012/climate_sigma4_v2",1],
             }

MODEL_INFO = {"PASR_ODE_small": {"lr": 1e-3,"batch_size": 16,"epochs": 500,"lr_step":40,"gamma":0.5},}

def generate_bash_script(data_name, model_name, scale_factor,seed=1234,method ="rk4",lamb_p= 0,upsampler="nearest_conv",n_snapshots=20,time_scale=1):
    job_name = f"{data_name}_{model_name}_{scale_factor}_{seed}_{method}_{lamb_p}_{upsampler}_{n_snapshots}_{time_scale}"
    short_name = f"{data_name}_{lamb_p}"
    if "FNO" in model_name:
        file = "train_FNO.py"
    else:
        file = "train.py"
    if lamb_p != 0:
        phyiscs = "True"
    else:
        phyiscs = "False"

    bash_content = f"""#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --account=dasrepo_g
#SBATCH --job-name={short_name}
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=Joey000122@gmail.com
#SBATCH --mail-type=ALL

module load pytorch/2.0.1

cmd1="srun python train.py --data_path {DATA_INFO[data_name][0]} --data {data_name} --in_channels {DATA_INFO[data_name][1]} --model {model_name} --lr {MODEL_INFO[model_name]['lr']} --batch_size {MODEL_INFO[model_name]['batch_size']} --epochs {MODEL_INFO[model_name]['epochs']} --lr_step {MODEL_INFO[model_name]['lr_step']} --gamma {MODEL_INFO[model_name]['gamma']} --seed {seed} --ode_method {method} --lamb_p {lamb_p} --physics {phyiscs} --timescale_factor {time_scale} --upsampler {upsampler} --n_snapshots {n_snapshots} --window_size 5"

set -x
bash -c "$cmd1"
"""

    with open(f"bash_script/{job_name}.sbatch", 'w') as out_file:
        out_file.write(bash_content)
        print(f"Bash script generated as {job_name}.sbatch")
    return  job_name
# Run the function
if __name__ == "__main__":
    data_name = ['era5']
    model_name =  "PASR_ODE_small"
    for name in data_name:
        for upsampler in ["nearest_conv","pixel_shuffle"]:
            for method in ["euler","rk4"]:
                    for n_snapshots in [5,10]:
                        for time_scale in [1,2]:
                            job_name = generate_bash_script(data_name=name,model_name=model_name,scale_factor=4,method=method,lamb_p=0.0,upsampler=upsampler,seed=3407,n_snapshots=n_snapshots,time_scale=time_scale)
                            with open("bash.sh","a") as f:
                                print(f"sbatch bash_script/{job_name}.sbatch",file=f)
                            f.close()
