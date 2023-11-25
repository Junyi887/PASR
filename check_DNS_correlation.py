import h5py
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
# f1 = h5py.File("/pscratch/sd/j/junyi012/jax_cfd_DNS/decay_turb_lres_sim_s2_1220.h5","r")
# f2 = h5py.File("/pscratch/sd/j/junyi012/jax_cfd_DNS/decay_turb_lres_sim_s4_1220.h5","r")
# f3 = h5py.File("/pscratch/sd/j/junyi012/jax_cfd_DNS/decay_turb_lres_sim_s8_1220.h5","r")
# f4 = h5py.File("/pscratch/sd/j/junyi012/jax_cfd_DNS/decay_turb_lres_sim_s16_1220.h5","r")
# f5 = h5py.File("/pscratch/sd/j/junyi012/jax_cfd_DNS/decay_turb_lres_sim_s32_1220.h5","r")

# w = f1["tasks"]["vorticity"][()]
# w_lr1 = f1["tasks"]["vorticity_lr"][()]
# w_lr2 = f2["tasks"]["vorticity_lr"][()]
# w_lr3 = f3["tasks"]["vorticity_lr"][()]
# w_lr4 = f4["tasks"]["vorticity_lr"][()]
# w_lr5 = f5["tasks"]["vorticity_lr"][()]

# t = f1["tasks"]["t"][:]
# correlations0 = np.zeros(w.shape[0])
# correlations1 = np.zeros(w.shape[0])
# correlations2 = np.zeros(w.shape[0])
# correlations3 = np.zeros(w.shape[0])
# correlations4 = np.zeros(w.shape[0])
# correlations5 = np.zeros(w.shape[0])
# for i in range(w.shape[0]):
#     correlations0[i],_ = pearsonr(w[i].flatten(),w[i].flatten())
#     correlations1[i],_ = pearsonr(w[i,::2,::2].flatten(),w_lr1[i].flatten())
#     correlations2[i],_ = pearsonr(w[i,::4,::4].flatten(),w_lr2[i].flatten())
#     correlations3[i],_ = pearsonr(w[i,::8,::8].flatten(),w_lr3[i].flatten())
#     correlations4[i],_ = pearsonr(w[i,::16,::16].flatten(),w_lr4[i].flatten())
#     correlations5[i],_ = pearsonr(w[i,::32,::32].flatten(),w_lr5[i].flatten())
# plt.figure()
# # Save the figure
# plt.plot(t,correlations0)
# plt.plot(t,correlations1)
# plt.plot(t,correlations2)
# plt.plot(t,correlations3)
# plt.plot(t,correlations4)
# plt.plot(t,correlations5)
# plt.legend(["s=1024","s=512","s=256","s=128","s=64","s=32"])
# plt.axhline(0.8,linestyle="--",color="grey",alpha=0.2)
# plt.xlabel("Time")
# plt.ylabel("Correlation")
# plt.ylim(0,1.1)
# ax = plt.gca()
# def forward(x):
#     return x // (t[2] - t[1])
# def inverse(x):
#     return x * (t[2] - t[1])

# # Adding a secondary x-axis on top for displaying the index of 't'
# ax2 = ax.secondary_xaxis('top', functions=(forward, inverse))

# print(t)
# print(len(t))
# # Optionally, set labels and ticks for ax2
# ax2.set_xlabel("Iteration step")
# plt.savefig("correlation2.png")

f01 = h5py.File("../DT_shorter/viz/decay_turb_lres_sim_s4_2250.h5","r")
t2 = f01["tasks"]["t"][()]
w_lr6 = f01["tasks"]["vorticity_lr"][()]
w_hr6 = f01["tasks"]["vorticity"][()]
correlations06 = np.zeros(w_hr6.shape[0])
correlations6 = np.zeros(w_hr6.shape[0])
for i in range(w_hr6.shape[0]):
    correlations06[i],_= pearsonr(w_hr6[i].flatten(),w_hr6[i].flatten())
    correlations6[i],_ = pearsonr(w_hr6[i,::4,::4].flatten(),w_lr6[i].flatten())
plt.figure()
# Save the figure
plt.plot(t2,correlations06)
plt.plot(t2,correlations6)

plt.legend(["s=128","s=32"])
plt.axhline(0.8,linestyle="--",color="grey",alpha=0.2)
plt.xlabel("Time")
plt.ylabel("Correlation")
plt.ylim(0,1.1)
ax = plt.gca()
def forward(x):
    return x // (t2[2] - t2[1])
def inverse(x):
    return x * (t2[2] - t2[1])

# Adding a secondary x-axis on top for displaying the index of 't'
ax2 = ax.secondary_xaxis('top', functions=(forward, inverse))
# Optionally, set labels and ticks for ax2
ax2.set_xlabel("Iteration step")
plt.savefig("correlation1.png")