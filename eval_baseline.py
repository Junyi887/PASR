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
from src.util import *
from src.models import *

FLUID_DATA_INFO = {"decay_turb":['../Decay_Turbulence_small/test/Decay_turb_small_128x128_323.h5', 0.02],
                 "burger2d": ["../Burgers_2D_small/test/Burgers2D_128x128_79.h5",0.001],
                 "rbc": ["../RBC_small/test/RBC_small_33_s2.h5",0.01]}

steps = 21 # 40 
effective_step = list(range(0, steps))
FLUID_DATA_INFO2 = {"decay_turb":"../Decay_Turbulence_small/*/*.h5",
                    "rbc": "../RBC_small/*/*.h5"}
def get_psnr(true, pred):
    # shape with B,T,C,H,W
    print("adfadfadf",true.shape,pred.shape)
    mse = torch.mean((true - pred)**2,dim= (-1,-2))
    if mse.min() <= 1e-16:
        return float('inf')
    B,T,C = mse.shape
    print(mse.shape)
    list = []
    for i in range(T):
        max_value = torch.max(true)
        psnr = 20 * torch.log10(max_value / torch.sqrt(mse[:,i,:])) # return psnr in shape B,C,T
        list.append(psnr)
    return torch.stack(list,dim =1) # return psnr in shape B,C,T

def get_ssim(true,pred):
    from torchmetrics import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure().cuda()
    B,T,C,H,W = true.shape
    true = true.reshape(B*T,C,H,W)
    pred = pred.reshape(B*T,C,H,W)
    list = []
    for i in range(C):
        ssim_score = ssim(pred[:,i:(i+1)],true[:,i:(i+1)])
        list.append(ssim_score) 
    list = torch.stack(list,dim =-1) # return ssim in shape B,C
    return list

def dispatch_ConvLSTM(data_name,in_channels,num_snapshots=20):
    if data_name !="climate":
        normalizer = DataInfoLoader(FLUID_DATA_INFO2[data_name])
        mean,std = normalizer.get_mean_std()
        print('mean of hres is:',mean.tolist())
        print('stf of hres is:', std.tolist())
        steps = num_snapshots+1# 40 
        effective_step = list(range(0, steps))
    
    model = PhySR(
            n_feats = 32,
            n_layers = [1, 2], # [n_convlstm, n_resblock]
            upscale_factor = [num_snapshots, 4], # [t_up, s_up]
            shift_mean_paras = [mean.tolist(), std.tolist()],  
            step = steps,
            in_channels = in_channels,
            effective_step = effective_step).cuda()
    return model

def eval_ConvLSTM(data_name,model_path,lr_input_tensor,hr_target_tensor):
    model = dispatch_ConvLSTM(data_name,3)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    init_state = get_init_state(
        batch_size = [lr_input_tensor.shape[0]], 
        hidden_channels = [32], 
        output_size = [[lr_input_tensor.shape[-2], lr_input_tensor.shape[-1]]], # 32, 32 
        mode = 'random')
    with torch.no_grad():
        lres, hres = lr_input_tensor.permute(2,0,1,3,4), hr_target_tensor.permute(2,0,1,3,4) # (b,c,t,h,w) -> (t,b,c,h,w)
        lres, hres = lres.float().cuda(), hres.float().cuda()
        pred = model(lres,init_state)
        RFNE = torch.norm((pred-hres),dim=(-1,-2))/torch.norm(hres,dim=(-1,-2))
        MSE = torch.mean((pred-hres)**2,dim=(-1,-2))
        MAE = torch.mean(torch.abs(pred-hres),dim=(-1,-2)) 
        # permute back:
        SSIM = get_ssim(hres.permute(1,0,2,3,4),pred.permute(1,0,2,3,4)).cpu().numpy() # from T,b,C,H,W to b,t,C,H,W
        PSNR = get_psnr(hres.permute(1,0,2,3,4),pred.permute(1,0,2,3,4)).cpu().numpy()
        RFNE = RFNE.permute(1,2,0).cpu().numpy()
        MSE = MSE.permute(1,2,0).cpu().numpy()
        MAE = MAE.permute(1,2,0).cpu().numpy()

        print(f"SSIM shape {SSIM.shape}")
        print(f"PSNR shape {PSNR.shape}")
        print(f"SSIM (ConvLSTM) {SSIM.mean():.4f} +/- {SSIM.std():.4f}")
        print(f"PSNR (ConvLSTM) {PSNR.mean():.4f} +/- {PSNR.std():.4f}")
        print("channel wise SSIM (ConvLSTM) ", SSIM.tolist())
        print("channel wise PSNR (ConvLSTM) ", PSNR.mean(axis=(0,1)).tolist())
    # print(f"(ConvLSTM) RFNE of third batch first channel {RFNE[2,0].mean(axis=0)}")
    # print(f"(ConvLSTM) MSE of third batch first channel {MSE[2,0].mean(axis=0)}")
    # print(f"(ConvLSTM) MAE of third batch first channel {MAE[2,0].mean(axis=0)}")
    # print(f"")
    return pred, RFNE, MSE, MAE

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
    PSNR = get_psnr(hr_target_tensor.permute(0,2,1,3,4),trilinear_pred.permute(0,2,1,3,4)).cpu().numpy()
    SSIM = get_ssim(hr_target_tensor.permute(0,2,1,3,4),trilinear_pred.permute(0,2,1,3,4)).cpu().numpy()
    print(f"SSIM (Trilinear) {SSIM.mean():.4f} +/- {SSIM.std():.4f}")
    print(f"PSNR (Trilinear) {PSNR.mean():.4f} +/- {PSNR.std():.4f}")
    print("channel wise SSIM (Trilinear) ", SSIM.tolist())
    print("channel wise PSNR (Trilinear) ", PSNR.mean(axis=(0,1)).tolist())
    print(f"Saved error shape {RFNE.shape}")
    # print(f"(Trilinear) RFNE of third batch first channel {RFNE[2,0].mean(dim=0)}")
    # print(f"(Trilinear) MSE of third batch first channel {MSE[2,0].mean(dim=0)}")
    # print(f"(Trilinear) MAE of third batch first channel {MAE[2,0].mean(dim=0)}")
    print(f"")
    # Finite differnece reconstruction
    return trilinear_pred.numpy(),RFNE.numpy(),MSE.numpy(),MAE.numpy()
# lr_input,hr_target,lr_input_tensor,hr_target_tensor =load_test_data_squence("decay_turb",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
# trilinear_interpolation(lr_input_tensor,hr_target_tensor)
# lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("rbc",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
# trilinear_interpolation(lr_input_tensor,hr_target_tensor)

if __name__ == "__main__":
    lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("decay_turb",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
    _,RFNE_tri,MSE_tri,MAE_tri = trilinear_interpolation(lr_input_tensor,hr_target_tensor)
    print(RFNE_tri.shape)
    print(RFNE_tri.mean(axis=(0,1)))
    np.save("DT_RFNE_trilinear_extrapolation.npy",RFNE_tri)
    np.save("DT_MSE_trilinear.npy",MSE_tri)
    np.save("DT_MAE_trilinear.npy",MAE_tri)

    print(f"RFNE (Trilinear): {RFNE_tri.mean():.4f} +/- {RFNE_tri.std():.4f}")
    print(f"MAE (Trilinear): {MAE_tri.mean():.4f} +/- {MAE_tri.std():.4f}")
    print("channel wise RFNE (Trilinear) ",RFNE_tri.mean(axis=(0,-1)).tolist())
    print("channel wise MAE (Trilinear) ",MAE_tri.mean(axis=(0,-1)).tolist())
    _,RFNE_conv,MSE_conv,MAE_conv = eval_ConvLSTM("decay_turb","results/ConvLSTM_Decay_Turb_checkpoint.pt",lr_input_tensor,hr_target_tensor)
    print(f"RFNE (ConvLSTM) {RFNE_conv.mean():.4f} +/- {RFNE_conv.std():.4f}")
    print(f"MAE (ConvLSTM) {MAE_conv.mean():.4f} +/- {MAE_conv.std():.4f}")
    print("channel wise RFNE (ConvLSTM) ", RFNE_conv.mean(axis=(0,-1)).tolist())
    print("channel wise MAE (ConvLSTM) ", MAE_conv.mean(axis=(0,-1)).tolist())

    np.save("DT_RFNE_convL.npy",RFNE_conv)
    np.save("DT_MSE_convL.npy",MSE_conv)
    np.save("DT_MAE_convL.npy",MAE_conv)

    lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("rbc",timescale_factor = 4,num_snapshot = 20,in_channel=3,upscale_factor=4)
    _,RFNE_tri,MSE_tri,MAE_tri = trilinear_interpolation(lr_input_tensor,hr_target_tensor)
    np.save("RBC_RFNE_trilinear.npy",RFNE_tri)
    np.save("RBC_MSE_trilinear.npy",MSE_tri)
    np.save("RBC_MAE_trilinear.npy",MAE_tri)
    print(f"RFNE (Trilinear): {RFNE_tri.mean():.4f} +/- {RFNE_tri.std():.4f}")
    print(f"MAE (Trilinear): {MAE_tri.mean():.4f} +/- {MAE_tri.std():.4f}")
    print("channel wise RFNE (Trilinear) ",RFNE_tri.mean(axis=(0,-1)).tolist())
    print("channel wise MAE (Trilinear) ",MAE_tri.mean(axis=(0,-1)).tolist())
    _,RFNE_conv,MSE_conv,MAE_conv = eval_ConvLSTM("rbc","results/ConvLSTM_RBC_checkpoint.pt",lr_input_tensor,hr_target_tensor)
    print(f"RFNE (ConvLSTM) {RFNE_conv.mean():.4f}  +/- {RFNE_conv.std():.4f}")
    print(f"MAE (ConvLSTM) {MAE_conv.mean():.4f} +/- {MAE_conv.std():.4f} ")
    print("channel wise RFNE (ConvLSTM) ", RFNE_conv.mean(axis=(0,-1)).tolist())
    print("channel wise MAE (ConvLSTM) ", MAE_conv.mean(axis=(0,-1)).tolist())
    np.save("RBC_RFNE_convL.npy",RFNE_conv)
    np.save("RBC_MSE_convL.npy",MSE_conv)
    np.save("RBC_MAE_convL.npy",MAE_conv)
    # lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data_squence("climate",timescale_factor = 10,num_snapshot = 10,in_channel=1,upscale_factor=4)
    # _,RFNE_tri,MSE_tri,MAE_tri = trilinear_interpolation(lr_input_tensor,hr_target_tensor)
    # np.save("Climate_RFNE_trilinear.npy",RFNE_tri)
    # np.save("Climate_MSE_trilinear.npy",MSE_tri)
    # np.save("Climate_MAE_trilinear.npy",MAE_tri)
    # print("all channel batch and time averaged (Trilinear)",RFNE_tri.mean())
