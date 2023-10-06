import h5py
import numpy as np
import torch
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
DATA_INFO = {"decay_turb":['../Decay_Turbulence_small/test/Decay_turb_small_128x128_323.h5', 0.02],
                 "burger2d": ["../Burgers_2D_small/test/Burgers2D_128x128_79.h5",0.001],
                 "rbc": ["../RBC_small/test/RBC_small_33_s2.h5",0.01]}

# def trilinear(file_path, crop, space_scale, time_scale):
#     # Window calculation

#     # Loading file and processing data
#     with h5py.File(file_path, 'r') as file:
#         train_hr = file['fields'][()]

#     x_lr_t_lr = train_hr[::time_scale, ::space_scale, ::space_scale]

#     # Preparing interpolation
#     x_lr_t_lr = torch.tensor(x_lr_t_lr).unsqueeze(1)
#     pred_x = F.interpolate(x_lr_t_lr, size=(crop, crop), mode='trilinear', align_corners=False).squeeze()
#     t_hr = np.linspace(0, 1, train_hr.shape[0])
#     t_lr = np.linspace(0, 1, x_lr_t_lr.shape[0])
#     cs = CubicSpline(t_lr, pred_x.numpy())
#     pred = torch.tensor(cs(t_hr)).unsqueeze(1)
#     train_poly_truth = torch.tensor(train_hr).unsqueeze(1)

#     # Error calculation
#     RFNE = torch.stack([
#         torch.norm(pred[i] - train_poly_truth[i]) / torch.norm(train_poly_truth[i])
#         for i in range(len(pred))
#         if i % time_scale != 0
#     ])

#     return torch.mean(RFNE)

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

def trilinear_interpolation(data_name,timescale_factor = 10,num_snapshot = 10,in_channel=1,upscale_factor=4):

    with h5py.File(DATA_INFO[data_name][0],'r') as f:
        w_truth = f['tasks']['vorticity'][()] if in_channel ==1 or in_channel ==3 else None
        u_truth = f['tasks']['u'][()]
        v_truth = f['tasks']['v'][()]
    final_index = (u_truth.shape[0]-1)//timescale_factor
    idx_matrix = generate_test_matrix(num_snapshot +1 , final_index)*timescale_factor    
    print(idx_matrix[1:3])
    if in_channel ==1:
        hr_input = w_truth[idx_matrix[:,0]]
        hr_target = w_truth[idx_matrix[:,:]]
    elif in_channel ==2:
        hr_input_in = np.stack((u_truth[idx_matrix[:,0]],v_truth[idx_matrix[:,0]]),axis=1)
        hr_input_end = np.stack((u_truth[idx_matrix[:,-1]],v_truth[idx_matrix[:,-1]]),axis=1)
        hr_input = np.stack((hr_input_in,hr_input_end),axis=2)
        print(f"hr input shape {hr_input.shape}")
        hr_target = np.stack((u_truth[idx_matrix[:,:]],v_truth[idx_matrix[:,:]]),axis=1)
    elif in_channel ==3:
        hr_input_in = np.stack((w_truth[idx_matrix[:,0]],u_truth[idx_matrix[:,0]],v_truth[idx_matrix[:,0]]),axis=1) # B,C,H,W
        hr_input_end = np.stack((w_truth[idx_matrix[:,0]],u_truth[idx_matrix[:,-1]],v_truth[idx_matrix[:,-1]]),axis=1)# B,C,H,W
        hr_input = np.stack((hr_input_in,hr_input_end),axis=2) #B,C,t,H,W
        print(f"hr input shape {hr_input.shape}")
        hr_target = np.stack((w_truth[idx_matrix[:,:]],u_truth[idx_matrix[:,:]],v_truth[idx_matrix[:,:]]),axis=1) #B,C,t,H,W
    print(f"hr target shape {hr_target.shape}")
    transform = torch.from_numpy
    B,C,t,H,W = hr_input.shape
    hr_input = hr_input.reshape((B,C*t,H,W))
    lr_input_tensor = F.interpolate(transform(hr_input),size = (int(H/upscale_factor),int(W/upscale_factor)),mode='bicubic',align_corners=False)
    lr_input_tensor = lr_input_tensor.reshape(B,C,t,int(H/upscale_factor),int(W/upscale_factor))
    print(f"lr input shape {lr_input_tensor.shape}")
    # List to store resized slices
    # resized_slices = []

    # # Iterate over the depth dimension
    # # for d in range(hr_input.shape[2]):
    # #     slice_d = transform(hr_input[:, :, d, :, :])
    # #     resized_slice = F.interpolate(slice_d, size=(32, 32), mode='bilinear', align_corners=False)
    # #     resized_slices.append(resized_slice)

    # # Stack resized slices along the depth dimension
    # lr_input_tensor = torch.stack(resized_slices, dim=2)
    
    trilinear_pred = F.interpolate(lr_input_tensor, size=(hr_target.shape[2],H,W), mode='trilinear', align_corners=False)
    print(f"trilinear pred shape {trilinear_pred.shape}")
    hr_target_tensor = transform(hr_target)
    lr_input_tensor = lr_input_tensor.unsqueeze(1) if in_channel ==1 else lr_input_tensor
    hr_target_tensor = hr_target_tensor.unsqueeze(2) if in_channel ==1 else hr_target_tensor
    lr_input = lr_input_tensor.numpy()
    RFNE = torch.norm((trilinear_pred-hr_target_tensor),dim=(-1,-2))/torch.norm(hr_target_tensor,dim=(-1,-2))
    print(RFNE.shape)
    
    print(RFNE[:,0].mean(dim=0))
    print(RFNE.mean().item())
    # Finite differnece reconstruction

    return lr_input,hr_target,lr_input_tensor,hr_target_tensor,trilinear_pred


trilinear_interpolation("decay_turb",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
# trilinear_interpolation("burger2d",timescale_factor = 4,num_snapshot = 20,in_channel=2,upscale_factor=4)
trilinear_interpolation("rbc",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
