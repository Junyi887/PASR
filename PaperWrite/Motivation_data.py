import matplotlib.pyplot as plt
import numpy as np

import h5py

file = h5py.File("/pscratch/sd/j/junyi012//DT_shorter/train/decay_turb_lres_sim_s4_2258.h5", 'r')
data = file['tasks']['vorticity'][:]
print(data.shape)
data_lr = file['tasks']['vorticity_lr'][:]
import torch
import torch.nn.functional as F
print(data_lr.shape)
data_bicubic_up = F.interpolate(torch.tensor(data_lr).unsqueeze(1),scale_factor=4,mode="bicubic",align_corners=False).squeeze().numpy()
# bicubic Downsample
data_lr = F.interpolate(torch.tensor(data).unsqueeze(1),scale_factor=0.25,mode="bicubic",align_corners=False).squeeze().numpy()
print(data_lr.shape)
print(data_bicubic_up.shape)

# get correlation 
from scipy.stats import pearsonr
from scipy.ndimage import zoom
correlation_x = np.zeros(data.shape[0])
correlation_t = np.zeros(data.shape[0])
for t in range (data.shape[0]):
    corr_x = pearsonr(data[t].flatten(),data_bicubic_up[t].flatten())
    correlation_x[t] = corr_x

