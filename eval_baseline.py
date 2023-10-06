import h5py
import numpy as np
import torch
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau 
from torch.nn.utils import weight_norm

from src.models import *
FLUID_DATA_INFO = {"decay_turb":['../Decay_Turbulence_small/test/Decay_turb_small_128x128_323.h5', 0.02],
                 "burger2d": ["../Burgers_2D_small/test/Burgers2D_128x128_79.h5",0.001],
                 "rbc": ["../RBC_small/test/RBC_small_33_s2.h5",0.01]}

def generate_test_matrix(cols:int, final_index:int):
    rows = (final_index + 1) // (cols - 1)
    if (final_index + 1) % (cols - 1) != 0:
        rows += 1
    matrix = np.zeros((rows, cols),dtype=int)
    current_value = 0
    for i in range(rows):
        for j in range(cols):
            if current_value <= final_index:
                matrix[i, j] = current_value
                current_value += 1
        current_value -= 1  # Repeat the last element in the next row
    return matrix[:-1,:]

def load_test_data_squence(data_name,in_channel,timescale_factor = 10,num_snapshot = 10,upscale_factor=4):
    if data_name == "climate":
        with h5py.File("/pscratch/sd/j/junyi012/climate_data/pre-processed_s4/climate_ds4_c1.h5",'r') as f:
            u_truth = f['fields'][()][-300:]
        final_index = (u_truth.shape[0]-1)//timescale_factor
        idx_matrix = generate_test_matrix(num_snapshot +1 , final_index)*timescale_factor
        hr_input_in = u_truth[idx_matrix[:,0]] # B,H,W
        hr_input_end = u_truth[idx_matrix[:,-1]]
        hr_target = u_truth[idx_matrix[:,:]] #B,T,H,W   
        hr_input_in = np.expand_dims(hr_input_in,axis=1) 
        hr_input_end = np.expand_dims(hr_input_end,axis=1)
        hr_target = np.expand_dims(hr_target,axis=1) # B,C,T,H,W
    else:
        with h5py.File(FLUID_DATA_INFO[data_name][0],'r') as f:
            w_truth = f['tasks']['vorticity'][()] if in_channel ==1 or in_channel ==3 else None
            u_truth = f['tasks']['u'][()]
            v_truth = f['tasks']['v'][()]
        final_index = (u_truth.shape[0]-1)//timescale_factor
        idx_matrix = generate_test_matrix(num_snapshot +1 , final_index)*timescale_factor    
        print(idx_matrix[1:3])
        if in_channel ==1:
            hr_input_in = w_truth[idx_matrix[:,0]]
            hr_input_end = w_truth[idx_matrix[:,-1]]
            hr_target = w_truth[idx_matrix[:,:]]
        elif in_channel ==2:
            hr_input_in = np.stack((u_truth[idx_matrix[:,0]],v_truth[idx_matrix[:,0]]),axis=1)
            hr_input_end = np.stack((u_truth[idx_matrix[:,-1]],v_truth[idx_matrix[:,-1]]),axis=1)
            hr_target = np.stack((u_truth[idx_matrix[:,:]],v_truth[idx_matrix[:,:]]),axis=1)
        elif in_channel ==3:
            hr_input_in = np.stack((w_truth[idx_matrix[:,0]],u_truth[idx_matrix[:,0]],v_truth[idx_matrix[:,0]]),axis=1) # B,C,H,W
            hr_input_end = np.stack((w_truth[idx_matrix[:,0]],u_truth[idx_matrix[:,-1]],v_truth[idx_matrix[:,-1]]),axis=1)# B,C,H,W
            hr_target = np.stack((w_truth[idx_matrix[:,:]],u_truth[idx_matrix[:,:]],v_truth[idx_matrix[:,:]]),axis=1) #B,C,t,H,W
    hr_input = np.stack((hr_input_in,hr_input_end),axis=2) #B,C,t,H,W
    transform = torch.from_numpy
    B,C,t,H,W = hr_input.shape
    hr_input = hr_input.reshape((B,C*t,H,W))
    lr_input_tensor = F.interpolate(transform(hr_input),size = (int(H/upscale_factor),int(W/upscale_factor)),mode='bicubic',align_corners=False)
    lr_input_tensor = lr_input_tensor.reshape(B,C,t,int(H/upscale_factor),int(W/upscale_factor))
    lr_input = lr_input_tensor.numpy()
    hr_target_tensor = torch.from_numpy(hr_target)
    print(f"lr input shape {lr_input.shape}")
    print(f"hr input shape {hr_input.shape}")
    print(f"hr target shape {hr_target.shape}")
    return lr_input,hr_target,lr_input_tensor,hr_target_tensor

def trilinear_interpolation(lr_input_tensor,hr_target_tensor):
    B,C,T,H,W = hr_target_tensor.shape
    trilinear_pred = F.interpolate(lr_input_tensor, size=(T,H,W), mode='trilinear', align_corners=False)
    print(f"trilinear pred shape {trilinear_pred.shape}")
    RFNE = torch.norm((trilinear_pred-hr_target_tensor),dim=(-1,-2))/torch.norm(hr_target_tensor,dim=(-1,-2))
    MSE = torch.mean((trilinear_pred-hr_target_tensor)**2,dim=(-1,-2))
    MAE = torch.mean(torch.abs(trilinear_pred-hr_target_tensor),dim=(-1,-2)) # result in B C T 
    print(f"Saved error shape {RFNE.shape}")
    print(f"RFNE of third batch first channel {RFNE[2,0].mean(dim=0)}")
    print(f"MSE of third batch first channel {MSE[2,0].mean(dim=0)}")
    print(f"MAE of third batch first channel {MAE[2,0].mean(dim=0)}")
    print(f"")
    # Finite differnece reconstruction
    return RFNE,MSE,MAE
# lr_input,hr_target,lr_input_tensor,hr_target_tensor =load_test_data_squence("decay_turb",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
# trilinear_interpolation(lr_input_tensor,hr_target_tensor)
# lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("rbc",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
# trilinear_interpolation(lr_input_tensor,hr_target_tensor)
lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("climate",timescale_factor = 4,num_snapshot = 20,in_channel=1,upscale_factor=4)
RFNE_tri,MSE_tri,MAE_tri = trilinear_interpolation(lr_input_tensor,hr_target_tensor)