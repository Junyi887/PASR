import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
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
    print(f"shape is {RFNE.shape}")
    return RFNE.detach().cpu().numpy(), MAE.detach().cpu().numpy(), MSE.detach().cpu().numpy(), IN.detach().cpu().numpy()

def eval_NODE_wrapper(model_path,num_snapshots=20,task_dt=1,result_normalization=False):
    checkpoint = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import argparse
    model_state = checkpoint['model_state_dict']
    config = checkpoint['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
    window_size = args.window_size
    stats_loader = DataInfoLoader(args.data_path+"/*/*.h5")
    def get_normalizer(args,stats_loader=stats_loader):
        if args.normalization == "True":
            mean, std = stats_loader.get_mean_std()
            min,max = stats_loader.get_min_max()
            if args.in_channels==1:
                mean,std = mean[0:1].tolist(),std[0:1].tolist()
                min,max = min[0:1].tolist(),max[0:1].tolist()
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
    img_x, img_y = stats_loader.get_shape()
    height = (img_x // args.scale_factor // window_size + 1) * window_size
    width = (img_y // args.scale_factor // window_size + 1) * window_size
    model = PASR_ODE(upscale=args.scale_factor, in_chans=args.in_channels, img_size=(height,width), window_size=window_size, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,num_ode_layers = args.ode_layer,ode_method = args.ode_method,ode_kernel_size = args.ode_kernel,ode_padding = args.ode_padding,aug_dim_t=args.aug_dim_t)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    from src.util.eval_util import get_psnr,get_ssim
    test1_loader,test2_loader,test3_loader = getData_test(upscale_factor = args.scale_factor, 
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size = 64, 
                                                      data_path = args.data_path,
                                                      num_snapshots = args.n_snapshots,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=args.in_channels,)
    for batch in test1_loader: # better be the val loader, need to modify datasets, but we are good for now.
        with torch.no_grad():
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            inputs = inputs[:,0,...]
            model.eval()
            out = model(inputs,task_dt = args.task_dt,n_snapshots = args.n_snapshots) 
            RFNE1,MAE1,MSE1,IN1 = get_metric_RFNE(target,out)
    PSNR1 = get_psnr(out,target)
    SSIM1 = get_ssim(out,target)
    for batch in test3_loader:
        with torch.no_grad():
            lr, target = batch[0].float().to(device), batch[1].float().to(device)
            inputs = lr[:,0,...]
            model.eval()
            out = model(inputs,task_dt = args.task_dt,n_snapshots = args.n_snapshots) 
            RFNE2,MAE2,MSE2,IN2 = get_metric_RFNE(target,out)
    PSNR2 = get_psnr(out,target)
    SSIM2 = get_ssim(out,target)
    return lr.cpu().numpy(),target.cpu().numpy(),out.cpu().numpy(), (RFNE1,RFNE2), (MSE1,MSE2), (MAE1,MAE2),(IN1,IN2),(SSIM1,SSIM2),(PSNR1,PSNR2)
path = {"nearestconv_20":"results/PASR_ODE_small_data_decay_turbulence_lrsim_v2_4098.pt",
        "pixelshuffle_20":"results/PASR_ODE_small_data_decay_turbulence_lrsim_v2_982.pt",
        "pixelshuffle_30":"results/PASR_ODE_small_data_decay_turbulence_lrsim_v2_5111.pt"
}

name1 = "nearestconv_20"
name2 = "pixelshuffle_20"
name_ls = [name1,name2]
for name in name_ls:
    input,target,pred,rfne,mse,mae,inf,ssim,psnr= eval_NODE_wrapper(path[name])
    print(f"RFNE(no-roll-out) {rfne[0][:].mean()}; single trajectory: {rfne[1][:].mean()}")

    import seaborn
    fig,ax = plt.subplots(3,5,figsize=(5,3))
    for i in range(10):
        for j in range(5):
            ax[0,j].imshow(input[i*5,j*4,0,...],cmap=seaborn.cm.icefire)
            ax[0,j].axis("off")
            ax[1,j].imshow(target[i*5,j*4,0,...],cmap=seaborn.cm.icefire)
            ax[1,j].axis("off")
            ax[2,j].imshow(pred[i*5,j*4,0,...],cmap=seaborn.cm.icefire)
            ax[2,j].axis("off")
        fig.savefig(f"result_{name}_{i}.png",bbox_inches='tight')

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(rfne[0].mean(axis=1)[..., 0], label="RFNE_w")
    ax[0].plot(rfne[0].mean(axis=1)[..., 1], label="RFNE_u")
    ax[0].plot(rfne[0].mean(axis=1)[..., 2], label="RFNE_v")
    ax[1].plot(rfne[1].mean(axis=1)[..., 0], label="RFNE_w")
    ax[1].plot(rfne[1].mean(axis=1)[..., 1], label="RFNE_u")
    ax[1].plot(rfne[1].mean(axis=1)[..., 2], label="RFNE_v")
    ax[0].legend()
    ax[1].legend()
    ax[1].set_ylim([0, 0.7])
    ax[1].set_xlabel("Time")  # Fixed typo: set_xaxis -> set_xlabel
    fig.savefig(f"rfne_{name}.png")
    plt.figure()
    plt.plot(mse[1].mean(axis=1)[...,0],label="MSE_vorticity")
    plt.plot(mse[1].mean(axis=1)[...,1],label="MSE_u")
    plt.plot(mse[1].mean(axis=1)[...,2],label="MSE_v")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("MSE")
    plt.ylim(0,1.4)
    plt.title("averaged MSE in time with IC at differnt time")
    plt.savefig(f"mse_{name}.png")
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