import argparse


DATA_INFO = {'decay_turbulence':["/pscratch/sd/j/junyi012/Decay_Turbulence_small/","3"],
             "rbc":["/pscratch/sd/j/junyi012/RBC_small/","3"],
             "burgers2D":["/pscratch/sd/j/junyi012/burger2D_10/","3"],
            "decay_turbulence_lrsim":["../decay_turb_lrsim_short4","3"],
            "decay_turbulence_lrsim_v2":["../DT_shorter","3"],
            "ns_lres_sim_s4":["/pscratch/sd/j/junyi012/NS_lrsim_s4","3"],
            "DT_lrsim_256_s4_v0":["/pscratch/sd/j/junyi012/DT_lrsim_256_s4_v0","3"],
            "DT_lrsim_512_s4_v0":["/pscratch/sd/j/junyi012/DT_lrsim_512_s4_v0","3"],
            "DT_lrsim_1024_s4_v0":["/pscratch/sd/j/junyi012/DT_lrsim_1024_s4_v0","3"],
             }

MODEL_INFO = {"PASR_ODE_small": {"lr": 1e-3,"batch_size": 16,"epochs": 500,"lr_step":100,"gamma":0.5},}

def generate_bash_script(data_name, model_name,seed=1234,method ="rk4",lamb_p= 0,upsampler="nearest_conv",n_snapshots=20,in_channels=1,timescale_factor=1):
    job_name = f"{data_name}_{model_name}_{timescale_factor}_{seed}_{method}_{lamb_p}_{upsampler}_{n_snapshots}"
    short_name = f"{data_name}_{lamb_p}"
    if "FNO" in model_name:
        file = "baseline_FNO.py"
    else:
        file = "train.py"
    if lamb_p != 0:
        phyiscs = "True"
    else:
        phyiscs = "False"
    batch = 4 if data_name == "DT_lrsim_1024_s4_v0" else 16
    bash_content = f"""#!/bin/bash
#SBATCH --time=23:00:00
#SBATCH --account=dasrepo_g
#SBATCH --job-name={short_name}
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

cmd1="srun python {file} --data_path {DATA_INFO[data_name][0]} --data {data_name} --in_channels {in_channels} --model {model_name} --lr {MODEL_INFO[model_name]['lr']} --batch_size {batch} --epochs {MODEL_INFO[model_name]['epochs']} --lr_step {MODEL_INFO[model_name]['lr_step']} --gamma {MODEL_INFO[model_name]['gamma']} --seed {seed} --ode_method {method} --lamb_p {lamb_p} --physics {phyiscs} --timescale_factor {timescale_factor} --upsampler {upsampler} --n_snapshots {n_snapshots}"

set -x
bash -c "$cmd1"
"""

    with open(f"bash_script/{job_name}.sbatch", 'w') as out_file:
        out_file.write(bash_content)
        print(f"Bash script generated as {job_name}.sbatch")
    return  job_name
# Run the function
if __name__ == "__main__":
    data_name = ['DT_lrsim_256_s4_v0','DT_lrsim_512_s4_v0','DT_lrsim_1024_s4_v0']
    model_name =  "PASR_ODE_small"
    for name in data_name:
        for method in ["euler"]:
            for lamb_p in [0]:
                for upsampler in ["pixelshuffle"]:
                    for time in [10]:
                        for n_snapshots in [20]:
                            job_name = generate_bash_script(data_name=name,model_name=model_name,method=method,lamb_p=lamb_p,upsampler=upsampler,seed=3407,n_snapshots=n_snapshots,timescale_factor=time,in_channels=3)
                            with open("run_NODE.sh","a") as f:
                                print(f"sbatch bash_script/{job_name}.sbatch",file=f)
                            f.close()
