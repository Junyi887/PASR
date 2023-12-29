import h5py
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
def get_correlation(w_lr,w_hr):
    correlations = np.zeros(w_hr.shape[0])
    for i in range(w_hr.shape[0]):
        correlations[i],_ = pearsonr(w_lr[i].flatten(),w_hr[i].flatten())
    return correlations


correlations = []
scale =1
for i in [1024,512,256,128,64]:
    data_path = f"/pscratch/sd/j/junyi012/DT_multi_resolution_unpaired/DT_lres_sim_res_{i}_2405.h5"
    data = h5py.File(data_path, 'r')
    if i ==  1024:
        w_hr = data['tasks']['vorticity'][:]
        t = data['tasks']['t'][:]
    w = data['tasks']['vorticity'][:]
    cor = get_correlation(w_hr[:,::scale,::scale],w)
    correlations.append(cor)
    scale = scale *2
    print(scale)
j = 0
baseline_palette = sns.color_palette('YlGnBu', n_colors=7)[1:]
for correlation in correlations:
    plt.plot(t,correlation,color=baseline_palette[-j])
    j+=1

plt.legend(["s=1024","s=512","s=256","s=128","s=64"])
plt.axhline(0.8,linestyle="--",color="grey",alpha=0.2)
plt.xlabel("Time")
plt.ylabel("Correlation")
plt.ylim(0,1.1)
ax = plt.gca()
t2 = np.arange(0,2000,1)
def forward(x):
    return x // (t[2] - t[1])
def inverse(x):
    return x * (t[2] - t[1])
#Adding a secondary x-axis on top for displaying the index of 't'
ax2 = ax.secondary_xaxis('top', functions=(forward, inverse))
# Optionally, set labels and ticks for ax2
ax2.set_xlabel("Iteration step")
plt.savefig("correlation2.png")