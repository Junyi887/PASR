import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from src.util.eval_util import get_psnr,get_ssim
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch.nn.init as init
from math import log10
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from os import listdir
from os.path import join
from scipy.io import loadmat
from tqdm import tqdm
import h5py
from src.models import *
from src.util import *
from src.data_loader import getData_test
import logging
import random
from src.util.eval_util import get_psnr,get_ssim
from src.util.util_data_processing import get_normalizer



    
def get_metric_stats_metric(truth,pred,mean=0,std=1):
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
    # print(f"averaged RFNE {avg_RFNE}")
    # print(f"cumulative in time RFNE {cum_RFNE}")
    # print(f"shape is {RFNE.shape}")
    return RFNE.detach().cpu().numpy(), MAE.detach().cpu().numpy(), MSE.detach().cpu().numpy(), IN.detach().cpu().numpy(),cum_RFNE.detach().cpu().numpy()


def dump_json(key, RFNE, MAE, MSE, IN, SSIM, PSNR,cum_RFNE):
    import json
    magic_batch = 0
    magic_batch_end = -1
    json_file = "eval_v4.json"
    # Check if the results file already exists and load it, otherwise initialize an empty list
    try:
        with open(json_file, "r") as f:
            all_results = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_results = {}
        print("No results file found, initializing a new one.")
    # Create a unique key based on your parameters
    # Check if the key already exists in the dictionary
    if key not in all_results:
        all_results[key] = {
        }
    # Store the results
    print(f"RFNE shape is {RFNE.shape}")
    print(f"RFNE cumRFNE {cum_RFNE.mean(axis=0)}")
    print(f"RFNE {RFNE[0,:,0]}")
    print(f"RFNE {RFNE.mean(axis=(0,-1))}")
    all_results[key]["RFNE"] = RFNE[magic_batch:magic_batch_end,:,0].mean().item()
    all_results[key]["MAE"] = MAE[magic_batch:magic_batch_end,:,0].mean().item()
    all_results[key]["MSE"] = MSE[magic_batch:magic_batch_end,:,0].mean().item()
    all_results[key]["IN"] = IN[magic_batch:magic_batch_end,:,0].mean().item()
    all_results[key]["RFNE_v2"] = cum_RFNE[magic_batch:magic_batch_end,0].mean().item()
    all_results[key]["SSIM"] = SSIM[:magic_batch_end].mean().item()
    all_results[key]["PSNR"] = PSNR[:magic_batch_end].mean().item()
    with open(json_file, "w") as f:
        json.dump(all_results, f, indent=4)
        f.close()
    return print("dump json done")



def process_loader_NODE(loader, model, device,task_dt=1, n_snapshots=20):
    RFNE, MAE, MSE, IN, cum_RFNE = [], [], [], [], []
    PSNR, SSIM = [], []
    for batch in loader:
        with torch.no_grad():
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.eval()
            out = model(inputs, task_dt=task_dt, n_snapshots=n_snapshots) # Return B,C,T,H,W
            rfne, mae, mse, inf, cum_rfne = get_metric_stats_metric(target, out)
            RFNE.append(rfne)
            MAE.append(mae)
            MSE.append(mse)
            IN.append(inf)
            cum_RFNE.append(cum_rfne)
            PSNR.append(get_psnr(out, target).cpu().numpy())
            SSIM.append(get_ssim(out, target).cpu().numpy())
    return np.concatenate(RFNE, axis=0), np.concatenate(MAE, axis=0), np.concatenate(MSE, axis=0), np.concatenate(IN, axis=0), np.concatenate(cum_RFNE, axis=0), np.concatenate(PSNR, axis=0), np.concatenate(SSIM, axis=0)

def process_loader_baselines(loader, model, device,task_dt=1, n_snapshots=20):
    RFNE, MAE, MSE, IN, cum_RFNE = [], [], [], [], []
    PSNR, SSIM = [], []
    for batch in loader:
        with torch.no_grad():
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.eval()
            out = model(inputs) # Return B,C,T,H,W
            target = target.permute(0,2,1,3,4)
            out = out.permute(0,2,1,3,4) # Change shape to match required shape for metric calculation
            rfne, mae, mse, inf, cum_rfne = get_metric_stats_metric(target, out)
            RFNE.append(rfne)
            MAE.append(mae)
            MSE.append(mse)
            IN.append(inf)
            cum_RFNE.append(cum_rfne)
            PSNR.append(get_psnr(out, target).cpu().numpy())
            SSIM.append(get_ssim(out, target).cpu().numpy())
    return np.concatenate(RFNE, axis=0), np.concatenate(MAE, axis=0), np.concatenate(MSE, axis=0), np.concatenate(IN, axis=0), np.concatenate(cum_RFNE, axis=0), np.concatenate(PSNR, axis=0), np.concatenate(SSIM, axis=0)

def eval_NODE(model_path,num_snapshots=20,task_dt=1,result_normalization=False):
    checkpoint = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import argparse
    model_state = checkpoint['model_state_dict']
    config = checkpoint['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
    window_size = args.window_size
    stats_loader = DataInfoLoader(args.data_path+"/*/*.h5",args.data)
    mean,std = get_normalizer(args,stats_loader)
    img_x, img_y = stats_loader.get_shape()
    height = (img_x // args.scale_factor // window_size + 1) * window_size
    width = (img_y // args.scale_factor // window_size + 1) * window_size
    if hasattr(args, "final_tanh"):
        model = PASR_ODE(upscale=args.scale_factor, in_chans=args.in_channels, img_size=(height,width), window_size=window_size, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,num_ode_layers = args.ode_layer,ode_method = args.ode_method,ode_kernel_size = args.ode_kernel,ode_padding = args.ode_padding,aug_dim_t=args.aug_dim_t,final_tanh=args.final_tanh)
    else:
        model = PASR_ODE(upscale=args.scale_factor, in_chans=args.in_channels, img_size=(height,width), window_size=window_size, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,num_ode_layers = args.ode_layer,ode_method = args.ode_method,ode_kernel_size = args.ode_kernel,ode_padding = args.ode_padding,aug_dim_t=args.aug_dim_t,final_tanh=args.final_tanh)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    
    test1_loader,test2_loader,test3_loader = getData_test(upscale_factor = args.scale_factor, 
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size = 16, 
                                                      data_path = args.data_path,
                                                      num_snapshots = 20,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=args.in_channels,)

    RFNE1, MAE1, MSE1, IN1, cum_RFNE1, PSNR1, SSIM1 = process_loader_NODE(test1_loader, model, device, n_snapshots=20,task_dt=1)
    RFNE2, MAE2, MSE2, IN2, cum_RFNE2, PSNR2, SSIM2 = process_loader_NODE(test2_loader, model, device, n_snapshots=20,task_dt=1)
    RFNE3, MAE3, MSE3, IN3, cum_RFNE3, PSNR3, SSIM3 = process_loader_NODE(test3_loader, model, device, n_snapshots=20,task_dt=1)

    # Process the last batch outputs (optional, depending on your needs)
    return [RFNE1,RFNE2,RFNE3], [MSE1,MSE2,MSE3], [MAE1,MAE2,MAE3], [IN1,IN2,IN3], [SSIM1,SSIM2,SSIM3], [PSNR1,PSNR2,PSNR3],[cum_RFNE1,cum_RFNE2,cum_RFNE3]

def eval_FNO(model_path,in_channels=3,batch_size=4):

    checkpoint = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import argparse
    model_state = checkpoint['model_state_dict']
    config = checkpoint['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
    if hasattr(args, "width") and args.data.startswith("DT_512") == False:
        width = args.width
    else:
        width = 64
    fc_dim = 64 # or 40 for climate
    layers = [width, width, width, width, width]
    modes1 = [8, 8, 8, 8]
    modes2 = [8, 8, 8, 8]
    modes3 = [8, 8, 8, 8]
    stats_loader = DataInfoLoader(args.data_path+"/*/*.h5",args.data)
    mean,std = get_normalizer(args,stats_loader)
    img_x, img_y = stats_loader.get_shape()
    model = FNO3D(modes1, modes2, modes3,(args.batch_size,args.in_channels,args.n_snapshots+1,img_x,img_y),width=width, fc_dim=fc_dim,layers=layers,in_dim=args.in_channels, out_dim=args.in_channels, act='gelu',mean=mean,std=std )
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    test1_loader,test2_loader,test3_loader = getData_test(upscale_factor = args.scale_factor, 
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size = batch_size, 
                                                      data_path = args.data_path,
                                                      num_snapshots = 20,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=args.in_channels,)
    RFNE1, MAE1, MSE1, IN1, cum_RFNE1, PSNR1, SSIM1 = process_loader_baselines(test1_loader, model, device, args)
    RFNE2, MAE2, MSE2, IN2, cum_RFNE2, PSNR2, SSIM2 = process_loader_baselines(test2_loader, model, device, args)
    RFNE3, MAE3, MSE3, IN3, cum_RFNE3, PSNR3, SSIM3 = process_loader_baselines(test3_loader, model, device, args)
    return [RFNE1,RFNE2,RFNE3], [MSE1,MSE2,MSE3], [MAE1,MAE2,MAE3], [IN1,IN2,IN3], [SSIM1,SSIM2,SSIM3], [PSNR1,PSNR2,PSNR3],[cum_RFNE1,cum_RFNE2,cum_RFNE3]

def eval_ConvLSTM(model_path,in_channels=3):
    checkpoint = torch.load(model_path)
    checkpoint512 =  torch.load("results/FNO_data_DT_512_s4_v0_sequenceLR_1.pt")
    checkpoint256 = torch.load("results/FNO_data_DT_256_s4_v0_sequenceLR_2508.pt")
    checkpoint1024 = torch.load("results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt")
    if model_path.split("_")[2]=="512":
        checkpoint_data = checkpoint512
    elif model_path.split("_")[2]=="256":
        checkpoint_data = checkpoint256
    elif model_path.split("_")[2]=="1024":
        checkpoint_data = checkpoint1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import argparse
    model_state = checkpoint['model_state_dict']
    config = checkpoint_data['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
    num_snapshots = 20
    # stats_loader = DataInfoLoader(args.data_path+"/*/*.h5",args.data) 
    stats_loader = DataInfoLoader("/pscratch/sd/j/junyi012/DT_lrsim_1024_s4_v0/*/*.h5",args.data) #/pscratch/sd/j/junyi012/DT_lrsim_1024_s4_v0
    mean,std = get_normalizer(args,stats_loader)
    img_x, img_y = stats_loader.get_shape()
    steps = num_snapshots + 1 
    effective_step = list(range(0, steps))
    model = PhySR(
        n_feats=32,
        n_layers=[1, 2],  # [n_convlstm, n_resblock]
        upscale_factor=[num_snapshots, 4],  # [t_up, s_up]
        shift_mean_paras=[mean, std],  
        step=steps,
        in_channels=in_channels,
        effective_step=effective_step
    )
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    test1_loader,test2_loader,test3_loader = getData_test(upscale_factor = args.scale_factor, 
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size =16, 
                                                      data_path = args.data_path,
                                                      num_snapshots = 20,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=3,)
    RFNE1, MAE1, MSE1, IN1, cum_RFNE1, PSNR1, SSIM1 = process_loader_baselines(test1_loader, model, device, args)
    RFNE2, MAE2, MSE2, IN2, cum_RFNE2, PSNR2, SSIM2 = process_loader_baselines(test2_loader, model, device, args)
    RFNE3, MAE3, MSE3, IN3, cum_RFNE3, PSNR3, SSIM3 = process_loader_baselines(test3_loader, model, device, args)
    return [RFNE1,RFNE2,RFNE3], [MSE1,MSE2,MSE3], [MAE1,MAE2,MAE3], [IN1,IN2,IN3], [SSIM1,SSIM2,SSIM3], [PSNR1,PSNR2,PSNR3],[cum_RFNE1,cum_RFNE2,cum_RFNE3]


def eval_TriLinear(model_path,in_channels=3,batch_size=16,n_snapshots=20):
    checkpoint = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import argparse
    config = checkpoint['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
    stats_loader = DataInfoLoader(args.data_path+"/*/*.h5",args.data)
    mean,std = get_normalizer(args,stats_loader)
    model = TriLinear(upscale_factor=args.scale_factor,num_snapshots=n_snapshots)
    model = torch.nn.DataParallel(model).to(device)
    test1_loader,test2_loader,test3_loader = getData_test(upscale_factor = args.scale_factor, 
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size = batch_size, 
                                                      data_path = args.data_path,
                                                      num_snapshots = n_snapshots,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=in_channels,)
    RFNE1, MAE1, MSE1, IN1, cum_RFNE1, PSNR1, SSIM1 = process_loader_baselines(test1_loader, model, device, args)
    RFNE2, MAE2, MSE2, IN2, cum_RFNE2, PSNR2, SSIM2 = process_loader_baselines(test2_loader, model, device, args)
    RFNE3, MAE3, MSE3, IN3, cum_RFNE3, PSNR3, SSIM3 = process_loader_baselines(test3_loader, model, device, args)
    return [RFNE1,RFNE2,RFNE3], [MSE1,MSE2,MSE3], [MAE1,MAE2,MAE3], [IN1,IN2,IN3], [SSIM1,SSIM2,SSIM3], [PSNR1,PSNR2,PSNR3],[cum_RFNE1,cum_RFNE2,cum_RFNE3]


if __name__ == "__main__":
    path_lr_sim = {
        "PASR_DT_LR_SIM_1024_s4":"results/PASR_ODE_small_data_DT_lrsim_1024_s4_v0_8137.pt",
        # "PASR_DT_LR_SIM_512_s4":"results/PASR_ODE_small_data_DT_lrsim_512_s4_v0_5019.pt",
        # "PASR_DT_LR_SIM_256_s4":"results/PASR_ODE_small_data_DT_lrsim_256_s4_v0_2557.pt",
        "ConvLSTM_DT_LR_SIM_1024_s4":"results/ConvLSTM_DT_1024_s4_v0_sequenceLR5733_checkpoint.pt",
        # "ConvLSTM_DT_LR_SIM_512_s4":"ConvLSTM_DT_512_s4_v0_sequenceLR3895_checkpoint.pt",
        # "ConvLSTM_DT_LR_SIM_256_s4":"ConvLSTM_DT_256_s4_v0_sequenceLR865_checkpoint.pt",
        # "FNO_DT_LR_SIM_1024_s4":"results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt",
        # "FNO_DT_LR_SIM_512_s4":"results/FNO_data_DT_512_s4_v0_sequenceLR_1.pt",
        # "FNO_DT_LR_SIM_256_s4":"results/FNO_data_DT_256_s4_v0_sequenceLR_2508.pt",
        # "TriLinear_DT_LR_SIM_512_s4":"results/FNO_data_DT_512_s4_v0_sequenceLR_1.pt",
        # "TriLinear_DT_LR_SIM_1024_s4":"results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt",
        # "TriLinear_DT_LR_SIM_256_s4":"results/FNO_data_DT_256_s4_v0_sequenceLR_2508.pt",
    }

for name,model_path in path_lr_sim.items():
    if name.startswith("PASR"):
        rfne,mse,mae,inf,ssim,psnr,cumRFNE= eval_NODE(model_path)
        print(f"RFNE (with-roll-out) {rfne[0][:].mean()}; RFNE(No-roll-out) {rfne[1][:].mean()}; RFNE (single trajectory): {rfne[2][:].mean()},cumulative RFNE {cumRFNE[1][:].mean()}")
    elif name.startswith("FNO"):
        rfne,mse,mae,inf,ssim,psnr,cumRFNE= eval_FNO(model_path)
        print(f"RFNE (with-roll-out) {rfne[0][:].mean()}; RFNE(No-roll-out) {rfne[1][:].mean()}; RFNE (single trajectory): {rfne[2][:].mean()},cumulative RFNE {cumRFNE[1][:].mean()}")
    elif name.startswith("ConvLSTM"):
        rfne,mse,mae,inf,ssim,psnr,cumRFNE= eval_ConvLSTM(model_path)
        print(f"RFNE (with-roll-out) {rfne[0][:].mean()}; RFNE(No-roll-out) {rfne[1][:].mean()}; RFNE (single trajectory): {rfne[2][:].mean()},cumulative RFNE {cumRFNE[1][:].mean()}")
    else:
        rfne,mse,mae,inf,ssim,psnr,cumRFNE= eval_TriLinear(model_path)
        print(f"RFNE (with-roll-out) {rfne[0][:].mean()}; RFNE(No-roll-out) {rfne[1][:].mean()}; RFNE (single trajectory): {rfne[2][:].mean()},cumulative RFNE {cumRFNE[1][:].mean()}")
    dump_factor = 1
    dump_json(name,rfne[dump_factor],mae[dump_factor],mse[dump_factor],inf[dump_factor],ssim[dump_factor],psnr[dump_factor],cumRFNE[dump_factor])


    # import seaborn
    # fig,ax = plt.subplots(3,5,figsize=(5,3))
    # for i in range(10):
    #     for j in range(5):
    #         ax[0,j].imshow(input[i*5,j*4,0,...],cmap=seaborn.cm.icefire)
    #         ax[0,j].axis("off")
    #         ax[1,j].imshow(target[i*5,j*4,0,...],cmap=seaborn.cm.icefire)
    #         ax[1,j].axis("off")
    #         ax[2,j].imshow(pred[i*5,j*4,0,...],cmap=seaborn.cm.icefire)
    #         ax[2,j].axis("off")
    #     fig.savefig(f"result_{name}_{i}.png",bbox_inches='tight')

    # import matplotlib.pyplot as plt

    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # ax[0].plot(rfne[0].mean(axis=1)[..., 0], label="RFNE_w")
    # ax[0].plot(rfne[0].mean(axis=1)[..., 1], label="RFNE_u")
    # ax[0].plot(rfne[0].mean(axis=1)[..., 2], label="RFNE_v")
    # ax[1].plot(rfne[1].mean(axis=1)[..., 0], label="RFNE_w")
    # ax[1].plot(rfne[1].mean(axis=1)[..., 1], label="RFNE_u")
    # ax[1].plot(rfne[1].mean(axis=1)[..., 2], label="RFNE_v")
    # ax[0].legend()
    # ax[1].legend()
    # ax[1].set_ylim([0, 0.7])
    # ax[1].set_xlabel("Time")  # Fixed typo: set_xaxis -> set_xlabel
    # fig.savefig(f"rfne_{name}.png")
    # plt.figure()
    # plt.plot(mse[1].mean(axis=1)[...,0],label="MSE_vorticity")
    # plt.plot(mse[1].mean(axis=1)[...,1],label="MSE_u")
    # plt.plot(mse[1].mean(axis=1)[...,2],label="MSE_v")
    # plt.legend()
    # plt.xlabel("time")
    # plt.ylabel("MSE")
    # plt.ylim(0,1.4)
    # plt.title("averaged MSE in time with IC at differnt time")
    # plt.savefig(f"mse_{name}.png")

# fig,ax = plt.subplots(1,2,figsize=(10,5))
# ax[0].plot(mae[0].mean(axis=0)[...,0],label="RFNE_w")
# ax[0].plot(mae[0].mean(axis=0)[...,1],label="RFNE_u")
# ax[0].plot(mae[0].mean(axis=0)[...,2],label="RFNE_v")
# ax[1].plot(mae[1].mean(axis=0)[...,0],label="RFNE_w")
# ax[1].plot(mae[1].mean(axis=0)[...,1],label="RFNE_u")
# ax[1].plot(mae[1].mean(axis=0)[...,2],label="RFNE_v")
# ax[0].legend()
# ax[1].legend()
# fig.savefig("mae.png")