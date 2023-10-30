from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import cmocean
import h5py
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from src.models import *
# import torchaudio
from scipy.stats import pearsonr
from scipy.ndimage import zoom
CMAP = cmocean.cm.balance
CMAP = seaborn.cm.icefire
import argparse
from src.util import *
import torch
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
def energy_specturm(u,v):
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

DATA_INFO = {"decay_turb":['../Decay_Turbulence_small/test/Decay_turb_small_128x128_79.h5', 0.02],
                 "burger2d": ["../Burgers_2D_small/test/Burgers2D_128x128_79.h5",0.001],
                 "rbc": ["../RBC_small/test/RBC_small_33_s2.h5",0.01],}

def load_test_data(data_name,timescale_factor = 10,num_snapshot = 10,in_channel=1,upscale_factor=4):
    if data_name != "climate":
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
        with h5py.File("/pscratch/sd/j/junyi012/climate_data/pre-processed_s4_sig1/climate_ds4_c1_sigma1.h5",'r') as f:
            u_truth = f['fields'][()][-300:,...]
        final_index = (u_truth.shape[0]-1)//timescale_factor
        idx_matrix = generate_test_matrix(num_snapshot +1 , final_index)*timescale_factor    
        hr_input = u_truth[idx_matrix[:,0]]
        hr_target = u_truth[idx_matrix[:,:]]
        transform = torch.from_numpy
        img_shape_x = hr_input.shape[-2]
        img_shape_y = hr_input.shape[-1]
        input_transform = transforms.Resize((int(img_shape_x/upscale_factor),int(img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False)
        lr_input_tensor = input_transform(transform(hr_input))
        hr_target_tensor = transform(hr_target)
        lr_input_tensor = lr_input_tensor.unsqueeze(1)
        hr_target_tensor = hr_target_tensor.unsqueeze(2)
        lr_input = lr_input_tensor.numpy()
        return lr_input,hr_target,lr_input_tensor,hr_target_tensor      

                 
def get_prediction(model,lr_input_tensor,hr_target_tensor,scale_factor,in_channels,task_dt,n_snapshots,ode_step):
    model.eval()
    with torch.no_grad():
        pred = model(lr_input_tensor.float().cuda(),n_snapshots = n_snapshots,task_dt=task_dt)
    return pred.detach().cpu()


def get_metric_RFNE(pred,truth):
    # input should be tensor
    print("predshape",pred.shape)
    RFNE = torch.norm(pred.float().cuda()- truth.float().cuda(), p=2, dim=(-1, -2)) / torch.norm(truth.float().cuda(), p=2, dim=(-1, -2))
    MAE = torch.mean(torch.abs(pred.float().cuda()- truth.float().cuda()), dim=(-1, -2))
    MSE = torch.mean((pred.float().cuda()- truth.float().cuda())**2, dim=(-1, -2))
    IN = torch.norm((pred.float().cuda()- truth.float().cuda()),p=np.inf, dim=(-1, -2))
    avg_RFNE = RFNE.mean().item()
    cum_RFNE = torch.norm(pred.flatten().float().cuda()-truth.flatten().float().cuda(),p=2)/torch.norm(truth.flatten().float().cuda(),p=2)
    print(f"averaged RFNE {avg_RFNE}")
    print(f"cumulative RFNE {cum_RFNE.item()}")
    return RFNE.detach().cpu().numpy(),MAE.detach().cpu().numpy(),MSE.detach().cpu().numpy(),IN.detach().cpu().numpy()

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

# def plot_energy_specturm_phyLoss(data_name,dic):
#     # pred_trilinear = 
#     realsize_truth, EK_avsphr_truth,result_dict_truth = energy_specturm(u_truth,v_truth)
#     realsize_pred, EK_avsphr_pred,result_dict_pred = energy_specturm(u_pred,v_pred)
#     realsize_pred_p, EK_avsphr_pred_p,result_dict_pred_p = energy_specturm(u_pred_p,v_pred_p)
#     fig= plt.figure(figsize=(5,5))
#     plt.title(f"Kinetic Energy Spectrum -- {data_name}")
#     plt.xlabel(r"k (wavenumber)")
#     plt.ylabel(r"TKE of the k$^{th}$ wavenumber")
#     print(realsize_truth)
#     plt.loglog(np.arange(0,realsize_truth),((EK_avsphr_truth[0:realsize_truth] )),'k',label = "truth")
#     plt.loglog(np.arange(0,realsize_pred),((EK_avsphr_pred[0:realsize_pred] )),'r',label = "pred")
#     plt.loglog(np.arange(0,realsize_pred_p),((EK_avsphr_pred_p[0:realsize_pred_p] )),'b',label = "pred (physics constraint)")
#     plt.legend()
#     fig.savefig(f"{data_name}_energy_specturm.png",dpi=300,bbox_inches='tight')
#     return print("energy specturm plot done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PASR')
    parser.add_argument("--model_path", type=str, default="results/pre_trained_PASR_MLP_small_data_Decay_turb_small_0.pt", help="path to model")
    parser.add_argument("--test_data_name", type=str, default="decay_turb", help="decay_turb, burger2d, rbc")
    parser.add_argument("--saved_name", type=str, default="DT_", help="decay_turb, burger2d, rbc")
    parsed_args = parser.parse_args()
    checkpoint = torch.load(parsed_args.model_path)
    model_state = checkpoint['model_state_dict']
    config = checkpoint['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def get_normalizer(args):
        if args.normalization == "True":
            stats_loader = DataInfoLoader(args.data_path+"/*/*.h5")
            mean, std = stats_loader.get_mean_std()
            min,max = stats_loader.get_min_max()
            if args.in_channels==1:
                mean,std = mean[0].tolist(),std[0].tolist()
                min,max = min[0].tolist(),max[0].tolist()
            elif args.in_channels==3:
                mean,std = mean.tolist(),std.tolist()
                min,max = min.tolist(),max.tolist()
            elif args.in_channels==2:
                mean,std = mean[1:].tolist(),std[1:].tolist()
                min,max = min[1:].tolist(),max[1:].tolist()
            if args.normalization_method =="minmax":
                return min,max
            if args.normalization_method =="meanstd":
                return mean,std
        else:
            mean, std = [0], [1]
            mean, std = mean * args.in_channels, std * args.in_channels
            return mean,std
        
    mean,std = get_normalizer(args)

    if args.data =="Decay_turb_small": 
        image = [128,128]
    elif args.data =="rbc_small":
        image = [256,64]
    elif args.data =="Burger2D_small":
        image = [128,128]
    model_list = {
            "PASR_ODE_small":PASR_ODE(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,num_ode_layers = args.ode_layer,ode_method = args.ode_method,ode_kernel_size = args.ode_kernel,ode_padding = args.ode_padding,aug_dim_t=args.aug_dim_t),
    }
    model = model_list[args.model]
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    # lr_input_m1,hr_target_m1,lr_input_tensor_m1,hr_target_tensor_m1 = load_test_data(parsed_args.test_data_name,timescale_factor=8,num_snapshot = 10,in_channel=3,upscale_factor=4)
    lr_input,hr_target,lr_input_tensor,hr_target_tensor = load_test_data(parsed_args.test_data_name,timescale_factor=4,num_snapshot = 20,in_channel=3,upscale_factor=4)
    # lr_input2,hr_target2,lr_input_tensor2,hr_target_tensor2 = load_test_data(parsed_args.test_data_name,timescale_factor=2,num_snapshot = 40,in_channel=3,upscale_factor=4)
    # lr_input3,hr_target3,lr_input_tensor3,hr_target_tensor3 = load_test_data(parsed_args.test_data_name,timescale_factor=1,num_snapshot = 80,in_channel=3,upscale_factor=4)
    print("input shape", lr_input.shape)
    # pred_m1 = get_prediction(model,lr_input_tensor_m1,hr_target_tensor_m1,scale_factor = 4,in_channels = args.in_channels,task_dt = args.task_dt*2,n_snapshots = 10,ode_step=args.ode_step*2)
    pred = get_prediction(model,lr_input_tensor,hr_target_tensor,scale_factor = 4,in_channels = args.in_channels,task_dt =1.0,n_snapshots = 20,ode_step=12)
    # pred2 = get_prediction(model,lr_input_tensor2,hr_target_tensor2,scale_factor = 4,in_channels = args.in_channels,task_dt = args.task_dt/2,n_snapshots = 40,ode_step=args.ode_step//2)
    # pred3 = get_prediction(model,lr_input_tensor2,hr_target_tensor2,scale_factor = 4,in_channels = args.in_channels,task_dt = args.task_dt/4,n_snapshots = 80,ode_step=args.ode_step//4)
    RFNE,MAE,MSE,IN = get_metric_RFNE(pred,hr_target_tensor)
    PSNR = get_psnr(pred[:].float().cuda(),hr_target_tensor[:].float().cuda())
    SSIM = get_ssim(pred[:].float().cuda(),hr_target_tensor[:].float().cuda())
    print(RFNE.shape)
    print(f"RFNE {RFNE[5:].mean():.4f} +/- {RFNE[5:].std():.4f}")
    print(f"MAE {MAE[5:].mean():.4f} +/- {MAE[5:].std():.4f}")
    print("Channel-wise RFNE " ,RFNE[5:].mean(axis =(0,1)))
    print("Channel-wise MAE " ,MAE[5:].mean(axis =(0,1)))
    print(f"SSIM {SSIM.mean():.4f} +/- {SSIM.std():.4f}")
    print(f"PSNR {PSNR.mean():.4f} +/- {PSNR.std():.4f}")
    print("channel wise SSIM ", SSIM.tolist())
    print("channel wise PSNR  ", PSNR.mean(axis=(0,1)).tolist())
 # wrap-eval result into json
    import json
    #
    magic_batch = 5
    # Check if the results file already exists and load it, otherwise initialize an empty list
    try:
        with open("eval.json", "r") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}
        print("No results file found, initializing a new one.")
    # Create a unique key based on your parameters
    key = f"{args.model}_{args.data}_{args.ode_method}"
    # Check if the key already exists in the dictionary
    if key not in all_results:
        all_results[key] = {
        }
    # Store the results
    all_results[key]["RFNE"] = RFNE.mean().item()
    all_results[key]["MAE"] = MAE.mean().item()
    all_results[key]["MSE"] = MSE.mean().item()
    all_results[key]["IN"] = IN.mean().item()
    all_results[key]["SSIM"] = SSIM.mean().item()
    all_results[key]["PSNR"] = PSNR.mean().item()
    with open("eval.json", "w") as f:
        json.dump(all_results, f, indent=4)