import h5py
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from src.models import *
def get_correlation(w_lr,w_hr):
    correlations = np.zeros(w_hr.shape[0])
    for i in range(w_hr.shape[0]):
        correlations[i],_ = pearsonr(w_lr[i].flatten(),w_hr[i].flatten())
    return correlations


correlations = []

for i,scale in zip([1024,256,128,64],[1,4,8,16]):
    data_path = f"/pscratch/sd/j/junyi012/DT_multi_resolution_unpaired/DT_lres_sim_res_{i}_1402.h5"
    data = h5py.File(data_path, 'r')
    if i ==  1024:
        w_hr = data['tasks']['vorticity'][:]
        t = data['tasks']['t'][:]
    w = data['tasks']['vorticity'][:]
    model = Bicubic(upscale_factor=scale).to("cuda")
    w_lr = model(torch.tensor(torch.from_numpy(w)).unsqueeze(1).to("cuda")).detach().cpu().numpy()
    print(w_lr.shape,w_hr.shape)
    w_lr = w_lr.squeeze(1)
    cor = get_correlation(w_hr,w_lr)
    correlations.append(cor)
np.save("DNS_correlations.npy",correlations)
j = 1
plt.figure(figsize=(3,3))
baseline_palette = sns.color_palette('YlGnBu', n_colors=5)[1:]
for correlation in correlations:
    plt.plot(t,correlation,color=baseline_palette[-j])
    j+=1

plt.legend([r"$1024^2$",r"$256^2$",r"$128^2$",r"$64^2$"],fontsize="11")
plt.axhline(0.8,linestyle="--",color="grey",alpha=0.2)
plt.xlabel("Time",fontsize="11")
plt.ylabel("Correlation",fontsize="11")
plt.ylim(0.6,1.01)
ax = plt.gca()
t2 = np.arange(0,2000,1)
def forward(x):
    return x // (t[2] - t[1])
def inverse(x):
    return x * (t[2] - t[1])
#Adding a secondary x-axis on top for displaying the index of 't'
ax2 = ax.secondary_xaxis('top', functions=(forward, inverse))
# Optionally, set labels and ticks for ax2
ax2.set_xlabel("Iteration step",fontsize="11")
plt.savefig("PaperWrite/correlation_DNS_v2.pdf",bbox_inches="tight")