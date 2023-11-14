import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.data_loader_nersc import getData

DATA_INFO = {"decay_turb":['/pscratch/sd/j/junyi012/Decay_Turbulence_small/test/Decay_turb_small_128x128_79.h5', 0.02],
                 "burger2D": ["../burger2D_10/test/Burger2D_716_s1.h5",0.001],
                 "rbc": ["/pscratch/sd/j/junyi012/RBC_small/test/RBC_small_33_s2.h5",0.01],
                 "climate_s4_sig1": ["/pscratch/sd/j/junyi012/climate_data/s4_sig1/",1],
                "climate_s4_sig0": ["/pscratch/sd/j/junyi012/climate_data/s4_sig0/",1],
                "climate_s4_sig2": ["/pscratch/sd/j/junyi012/climate_data/s4_sig2/",1],
                "climate_s4_sig4": ["/pscratch/sd/j/junyi012/climate_data/s4_sig4/",1],
                 "climate_s2_sig0": ["/pscratch/sd/j/junyi012/climate_data/s2_sig0/",1],
                  "climate_s2_sig1": ["/pscratch/sd/j/junyi012/climate_data/s2_sig1/",1],
                   "climate_s2_sig2": ["/pscratch/sd/j/junyi012/climate_data/s2_sig2/",1],
                    "climate_s2_sig4": ["/pscratch/sd/j/junyi012/climate_data/s2_sig4/",1]}


def energy_specturm(u,v):
    """
    Computes the energy spectrum of the given velocity fields.

    Args:
    u (numpy.ndarray): in shape (T,H,W)
    v (numpy.ndarray): in shape (T,H,W)

    Returns:
    tuple: A tuple containing:
        - int: The maximum real K value.
        - numpy.ndarray: The energy spectrum of the velocity fields.
        - dict: A dictionary containing the following keys:
            - "Real Kmax": The maximum real K value.
            - "Spherical Kmax": The maximum spherical K value.
            - "KE of the mean velocity discrete": The kinetic energy of the mean velocity (discrete).
            - "KE of the mean velocity sphere": The kinetic energy of the mean velocity (spherical).
            - "Mean KE discrete": The mean kinetic energy (discrete).
            - "Mean KE sphere": The mean kinetic energy (spherical).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from math import sqrt
    data = np.stack((u,v),axis=1)
    print ("shape of data = ",data.shape)
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Reading files...localtime",localtime, "- END\n")
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Computing spectrum... ",localtime)
    N = data.shape[-1]
    M= data.shape[-2]
    print("N =",N)
    print("M =",M)
    eps = 1e-16 # to void log(0)
    U = data[:,0].mean(axis=0)
    V = data[:,1].mean(axis=0)
    amplsU = abs(np.fft.fftn(U)/U.size)
    amplsV = abs(np.fft.fftn(V)/V.size)
    print(f"amplsU.shape = {amplsU.shape}")
    EK_U  = amplsU**2
    EK_V  = amplsV**2 
    EK_U = np.fft.fftshift(EK_U)
    EK_V = np.fft.fftshift(EK_V)
    sign_sizex = np.shape(EK_U)[0]
    sign_sizey = np.shape(EK_U)[1]
    box_sidex = sign_sizex
    box_sidey = sign_sizey
    box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)
    centerx = int(box_sidex/2)
    centery = int(box_sidey/2)
    print ("box sidex     =",box_sidex) 
    print ("box sidey     =",box_sidey) 
    print ("sphere radius =",box_radius )
    print ("centerbox     =",centerx)
    print ("centerboy     =",centery)
    EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    EK_V_avsphr = np.zeros(box_radius,)+eps ## size of the radius
    for i in range(box_sidex):
        for j in range(box_sidey):          
            wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
            EK_U_avsphr[wn] = EK_U_avsphr [wn] + EK_U [i,j]
            EK_V_avsphr[wn] = EK_V_avsphr [wn] + EK_V [i,j]     
    EK_avsphr = 0.5*(EK_U_avsphr + EK_V_avsphr)
    realsize = len(np.fft.rfft(U[:,0]))
    TKEofmean_discrete = 0.5*(np.sum(U/U.size)**2+np.sum(V/V.size)**2)
    TKEofmean_sphere   = EK_avsphr[0]
    total_TKE_discrete = np.sum(0.5*(U**2+V**2))/(N*M) # average over whole domaon / divied by total pixel-value
    total_TKE_sphere   = np.sum(EK_avsphr)
    result_dict = {
    "Real Kmax": realsize,
    "Spherical Kmax": len(EK_avsphr),
    "KE of the mean velocity discrete": TKEofmean_discrete,
    "KE of the mean velocity sphere": TKEofmean_sphere,
    "Mean KE discrete": total_TKE_discrete,
    "Mean KE sphere": total_TKE_sphere
    }
    print(result_dict)
    localtime = time.asctime( time.localtime(time.time()) )
    print ("Computing spectrum... ",localtime, "- END \n")
    return realsize, EK_avsphr,result_dict

def get_psnr(true, pred,mean=0,std=1):
    # shape with B,T,C,H,W
    print("adfadfadf",true.shape,pred.shape)
    with torch.no_grad():
        if mean != 0:
            true = (true-mean)/std
            pred = (pred-mean)/std
        mse = torch.mean((true - pred)**2,dim= (-1,-2))
        if mse.min() <= 1e-16:
            return float('inf')
        B,T,C = mse.shape
        print(f"MSE shape in get_psnr: {mse.shape}")
        list = []
        for i in range(T):
            max_value = torch.max(true)
            psnr = 20 * torch.log10(max_value / torch.sqrt(mse[:,i,:])) # return psnr in shape B,C,T
            list.append(psnr)
    return torch.stack(list,dim =1) # return psnr in shape B,C,T

def get_ssim(true,pred,mean=0,std=1):
    from torchmetrics import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure().cuda()
    with torch.no_grad():
        B,T,C,H,W = true.shape
        true = true.reshape(B*T,C,H,W)
        pred = pred.reshape(B*T,C,H,W)
        if mean != 0:
            true = (true-mean)/std
            pred = (pred-mean)/std
        list = []
        for i in range(C):
            ssim_score = ssim(pred[:,i:(i+1)],true[:,i:(i+1)])
            list.append(ssim_score) 
        list = torch.stack(list,dim =-1) # return ssim in shape B,C
    return list

def get_metric_RFNE(truth,pred,mean=0,std=1):
    """
    Computes the Relative Frame-wise Norm Error (RFNE), Mean Absolute Error (MAE), Mean Squared Error (MSE),
    and Infinity Norm (IN) between the predicted and ground truth tensors.

    Args:
        truth (torch.Tensor): The ground truth tensor with shape B,T,C,H,W.
        pred (torch.Tensor): The predicted tensor with shape B,T,C,H,W.
        mean (float): The mean value used for normalization. Default is 0.
        std (float): The standard deviation value used for normalization. Default is 1.

    Returns:
        Tuple of numpy arrays containing the RFNE, MAE, MSE, and IN values.
    """
    # input should be tensor with shape B,T,C,H,W
    if mean != 0:
        pred = (pred-mean)/std
        truth = (truth-mean)/std
    RFNE = torch.norm(pred - truth, p=2, dim=(-1, -2)) / torch.norm(truth, p=2, dim=(-1, -2))
    MAE = torch.mean(torch.abs(pred - truth), dim=(-1, -2))
    MSE = torch.mean((pred - truth)**2, dim=(-1, -2))
    IN = torch.norm(pred - truth, p=np.inf, dim=(-1, -2))
    avg_RFNE = RFNE.mean().item()
    cum_RFNE = torch.norm(pred - truth, p=2, dim=(1,-1,-2)) / torch.norm(truth, p=2, dim=(1,-1,-2))
    print(f"averaged RFNE {avg_RFNE}")
    print(f"cumulative in time RFNE {cum_RFNE}")
    return RFNE.detach().cpu().numpy(), MAE.detach().cpu().numpy(), MSE.detach().cpu().numpy(), IN.detach().cpu().numpy()

def load_test_data_IC(data_name,timescale_factor = 4,num_snapshot = 10,in_channel=1,upscale_factor=4):
    if "climate" not in data_name:
        with h5py.File(DATA_INFO[data_name][0],'r') as f:
            w_truth = f['tasks']['vorticity'][()] if in_channel ==1 or in_channel ==3 else None
            u_truth = f['tasks']['u'][()]
            v_truth = f['tasks']['v'][()]
        final_index = (u_truth.shape[0]-1)//timescale_factor
        idx_matrix = generate_test_matrix(num_snapshot +1 , final_index)*timescale_factor    
        print(idx_matrix[0:2])
        if in_channel ==1:
            hr_input = w_truth[idx_matrix[:,0]]
            hr_target = w_truth[idx_matrix[:,:]]
        elif in_channel ==2:
            hr_input = np.stack((u_truth[idx_matrix[:,0]],v_truth[idx_matrix[:,0]]),axis=1)
            hr_target = np.stack((u_truth[idx_matrix[:,:]],v_truth[idx_matrix[:,:]]),axis=2)
        elif in_channel ==3:
            hr_input = np.stack((w_truth[idx_matrix[:,0]],u_truth[idx_matrix[:,0]],v_truth[idx_matrix[:,0]]),axis=1)
            hr_target = np.stack((w_truth[idx_matrix[:,:]],u_truth[idx_matrix[:,:]],v_truth[idx_matrix[:,:]]),axis=2)
        print(hr_target.shape)
        transform = torch.from_numpy
        img_shape_x = hr_input.shape[-2]
        img_shape_y = hr_input.shape[-1]
        input_transform = transforms.Resize((int(img_shape_x/upscale_factor),int(img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False)
        lr_input_tensor = input_transform(transform(hr_input))
        hr_target_tensor = transform(hr_target)
        lr_input_tensor = lr_input_tensor.unsqueeze(1) if in_channel ==1 else lr_input_tensor
        hr_target_tensor = hr_target_tensor.unsqueeze(2) if in_channel ==1 else hr_target_tensor
        lr_input = lr_input_tensor.numpy()
        return lr_input,hr_target,lr_input_tensor,hr_target_tensor
    else:
        _, _, _, _, test_loader = getData(upscale_factor=upscale_factor, 
                                           timescale_factor=timescale_factor,
                                           batch_size=12, 
                                           crop_size=720,
                                           data_path=DATA_INFO[data_name][0],
                                           num_snapshots=num_snapshot,
                                           noise_ratio=0.0,
                                           data_name="climate",
                                           in_channels=1,)
        for lr_input,hr_target in test_loader:
            lr_input_tensor = lr_input
            hr_target_tensor = hr_target
            break
        return lr_input_tensor.numpy(), hr_target_tensor.numpy(), lr_input_tensor, hr_target_tensor    
    return None

    
def load_test_data_sequence(data_name, in_channel=3, timescale_factor=4, num_snapshot=20, upscale_factor=4):
    """
    Load test data sequence for evaluation.

    Args:
        data_name (str): Name of the dataset.
        data_path (str): Path to the dataset.
        in_channel (int): Number of input channels.
        timescale_factor (int, optional): Factor by which to scale the time dimension. Defaults to 4.
        num_snapshot (int, optional): Number of snapshots to use. Defaults to 20.
        upscale_factor (int, optional): Factor by which to upscale the image. Defaults to 4.

    Returns:
        tuple: Tuple containing lr_input, hr_target, lr_input_tensor, and hr_target_tensor.
    """
    if "climate" in data_name:
        _, _, _, _, test_loader = getData(upscale_factor=upscale_factor, 
                                           timescale_factor=timescale_factor,
                                           batch_size=12, 
                                           crop_size=720,
                                           data_path=DATA_INFO[data_name][0],
                                           num_snapshots=num_snapshot,
                                           noise_ratio=0.0,
                                           data_name="climate_sequence",
                                           in_channels=1,)
        for lr_input,hr_target in test_loader:
            lr_input_tensor = lr_input
            hr_target_tensor = hr_target
            break
        return lr_input_tensor.numpy(), hr_target_tensor.numpy(), lr_input_tensor, hr_target_tensor
    else:
        with h5py.File(DATA_INFO[data_name][0], 'r') as f:
            w_truth = f['tasks']['vorticity'][()] if in_channel == 1 or in_channel == 3 else None
            u_truth = f['tasks']['u'][()]
            v_truth = f['tasks']['v'][()]
        final_index = (u_truth.shape[0]-1) // timescale_factor
        idx_matrix = generate_test_matrix(num_snapshot + 1, final_index) * timescale_factor    
        print(idx_matrix[1:3])
        if in_channel == 1:
            hr_input_in = w_truth[idx_matrix[:, 0]]
            hr_input_end = w_truth[idx_matrix[:, -1]]
            hr_target = w_truth[idx_matrix[:, :]]
        elif in_channel == 2:
            hr_input_in = np.stack((u_truth[idx_matrix[:, 0]], v_truth[idx_matrix[:, 0]]), axis=1)
            hr_input_end = np.stack((u_truth[idx_matrix[:, -1]], v_truth[idx_matrix[:, -1]]), axis=1)
            hr_target = np.stack((u_truth[idx_matrix[:, :]], v_truth[idx_matrix[:, :]]), axis=1)
        elif in_channel == 3:
            hr_input_in = np.stack((w_truth[idx_matrix[:, 0]], u_truth[idx_matrix[:, 0]], v_truth[idx_matrix[:, 0]]), axis=1) # B,C,H,W
            hr_input_end = np.stack((w_truth[idx_matrix[:, 0]], u_truth[idx_matrix[:, -1]], v_truth[idx_matrix[:, -1]]), axis=1)# B,C,H,W
            hr_target = np.stack((w_truth[idx_matrix[:, :]], u_truth[idx_matrix[:, :]], v_truth[idx_matrix[:, :]]), axis=1) #B,C,t,H,W
        hr_input = np.stack((hr_input_in, hr_input_end), axis=2) #B,C,t,H,W
        transform = torch.from_numpy
        B, C, t, H, W = hr_input.shape
        hr_input = hr_input.reshape((B, C*t, H, W))
        lr_input_tensor = F.interpolate(transform(hr_input), size=(int(H/upscale_factor), int(W/upscale_factor)), mode='bicubic', align_corners=False)
        lr_input_tensor = lr_input_tensor.reshape(B, C, t, int(H/upscale_factor), int(W/upscale_factor))
        lr_input = lr_input_tensor.numpy()
        hr_target_tensor = torch.from_numpy(hr_target)
        print(f"lr input shape {lr_input.shape}")
        print(f"hr input shape {hr_input.shape}")
        print(f"hr target shape {hr_target.shape}")
        return lr_input, hr_target, lr_input_tensor, hr_target_tensor
    return print("Wrong data name")

def generate_test_matrix(cols:int, final_index:int) -> np.ndarray:
    """
    Generates a test matrix with the given number of columns and final index.

    Args:
        cols (int): The number of columns in the matrix.
        final_index (int): The final index of the matrix.

    Returns:
        np.ndarray: A numpy array representing the generated matrix.
    """
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
