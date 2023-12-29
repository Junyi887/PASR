import h5py
import numpy as np
import matplotlib.pyplot as plt

data_path = "/pscratch/sd/j/junyi012/DT_multi_resolution_unpaired/DT_lres_sim_res_128_2405.h5"
data = h5py.File(data_path, 'r')

w_hr = data['tasks']['vorticity'][::10]
t = data['tasks']['t'][:]
w_lr = w_hr[:,::4,::4]

import seaborn as sns
for i in range(0,20):
    if i >=4 and i <=12:
        fig = plt.figure(figsize=(10,10))
        plt.imshow(w_lr[i*5],cmap=sns.cm.icefire)
        plt.axis('off')
        plt.savefig(f"PaperWrite/paper_figures/vorticity_lr_{i*5}.pdf",bbox_inches='tight',transparent=True)
        plt.close()
        fig = plt.figure(figsize=(10,10))
        plt.imshow(w_hr[i*5],cmap=sns.cm.icefire)
        plt.axis('off')
        plt.savefig(f"PaperWrite/paper_figures/vorticity_hr_{i*5}.pdf",bbox_inches='tight',transparent=True)
        plt.close()