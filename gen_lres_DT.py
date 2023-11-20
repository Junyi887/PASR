import random
import numpy as np

seed_list = [random.randint(0, 1000) for i in range(20)]
for seed in seed_list:
    with open('generate_DT_lrsim.sh',"a") as f:
        print(f"python data_gen/data_gen_DT.py --seed {seed};",file=f)
