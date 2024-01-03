import random
import numpy as np

seed_list = [random.randint(0, 5000) for i in range(20)]
for seed in seed_list:
    for res in [1024, 512, 256, 128, 64, 32]:
        with open('generate_DT_lrsim_v2.sh',"a") as f:
            print(f"python data_gen/generate_lres_DT_v2.py --seed {seed} --lr_res {res};",file=f)
