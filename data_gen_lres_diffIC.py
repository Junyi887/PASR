import random
import numpy as np

seed_list = [random.randint(0, 5000) for i in range(40)]
for seed in seed_list:
    with open('generate_NS_lrsim.sh',"a") as f:
        print(f"python data_gen/data_gen_Kolmogrove.py --seed {seed};",file=f)
