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
from src.util.eval_util import *
from src.util.util_data_processing import get_normalizer,DataInfoLoader
import logging
import random


def eval_NODE_correlation(model_path,num_snapshots=20,task_dt=1,result_normalization=False,timescale_factor=5):
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
        model = PASR_ODE(upscale=args.scale_factor, in_chans=args.in_channels, img_size=(height,width), window_size=window_size, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,num_ode_layers = args.ode_layer,ode_method = args.ode_method,ode_kernel_size = args.ode_kernel,ode_padding = args.ode_padding,aug_dim_t=args.aug_dim_t)
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    
    test1_loader,test2_loader,test3_loader = getData_test(upscale_factor = args.scale_factor, 
                                                      timescale_factor=timescale_factor,
                                                      batch_size = 32, 
                                                      data_path = args.data_path,
                                                      num_snapshots = num_snapshots,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=args.in_channels,)

    inputs,targets,preds =process_loader_NODE_viz(test3_loader, model, device,task_dt=task_dt, n_snapshots=num_snapshots)
    # Process the last batch outputs (optional, depending on your needs)
    return inputs,targets, preds

if __name__ == "__main__":
    path_lr_euler = {
        "PASR_DT_lrsim_1024_s4_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s4_v0_1537.pt",
        "PASR_DT_lrsim_1024_s8_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s8_v0_7228.pt",
        "PASR_DT_lrsim_1024_s16_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s16_v0_5143.pt",
    }
    path_lr_rk4 = {
        "PASR_DT_lrsim_1024_s4_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s4_v0_8137.pt",
        "PASR_DT_lrsim_1024_s8_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s8_v0_9438.pt",
        "PASR_DT_lrsim_1024_s16_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s16_v0_4342.pt",
    }
    
    # err_euler = []
    # err_rk4 = []
    label = [r"$\times 4$",r"$\times 8$",r"$\times 16$"]
    xlabel = [r"$\frac{\Delta t}{8}$",r"$\frac{\Delta t}{4}$",r"$\frac{\Delta t}{2}$",r"$\Delta t$",]
    # from scipy.stats import pearsonr
    # for name,model_path in path_lr_euler.items():
    #     local_err = []
    #     for n_snapshot,task_dt,timescale,in zip([200,100,40,20],[1.0,1.0,1.0,1.0],[10,5,2,1]):
    #         inputs,targets,preds = eval_NODE_correlation(model_path,num_snapshots=n_snapshot,task_dt=task_dt,result_normalization=False,timescale_factor=10//timescale)
    #         relative_error = np.linalg.norm(targets[:]-preds[:],axis=(-1,-2))/np.linalg.norm(targets[:],axis=(-1,-2))
    #         local_err.append(relative_error.mean())
    #     err_euler.append(local_err)

    # for name,model_path in path_lr_rk4.items():
    #     local_err = []
    #     for n_snapshot,task_dt,timescale,in zip([200,100,40,20],[1.0,1.0,1.0,1.0],[10,5,2,1]):
    #         inputs,targets,preds = eval_NODE_correlation(model_path,num_snapshots=n_snapshot,task_dt=task_dt,result_normalization=False,timescale_factor=10//timescale)
    #         relative_error = np.linalg.norm(targets[:]-preds[:],axis=(-1,-2))/np.linalg.norm(targets[:],axis=(-1,-2))
    #         local_err.append(relative_error.mean())
    #     err_rk4.append(local_err)

    # np.save("err_euler.npy",err_euler)
    # np.save("err_rk4.npy",err_rk4)
    err_euler = np.load("err_euler.npy")
    err_rk4 = np.load("err_rk4.npy")

    fig = plt.figure(figsize=(5,5))
    import seaborn as sns
    color_platte_euler = sns.color_palette("YlGnBu_r", 3)
    color_platte_rk4 = sns.color_palette("YlOrBr_r", 3)
    for i in range (len(err_euler)):
        plt.plot(range(len(xlabel)),err_euler[i]/err_euler[i][-1],'-o',color=color_platte_euler[i],label=f"Euler at {label[i]}")
    for i in range (len(err_rk4)):
        plt.plot(range(len(xlabel)),err_rk4[i]/err_rk4[i][-1],'-o',color=color_platte_rk4[i],label=f"rk4 at {label[i]}")
    plt.legend(fontsize=11)
    plt.xticks(range(len(xlabel)),xlabel,fontsize=11)
    plt.yscale("log")
    plt.yticks(fontsize=11)
    plt.ylabel("Relative error",fontsize=11)
    plt.xlabel("Time step",fontsize=11)
    plt.savefig("relative_error_convergence.pdf",bbox_inches='tight')

# def plot_vorticity_correlation(data_name,folder_name ="4090_results/"):
#     # data_truth = np.load(f"{folder_name}hr_target_{data_name}.npy")
#     # pred_tri = np.load(f"{folder_name}pred_tri_{data_name}.npy")
#     # pred_convL = np.load(f"{folder_name}pred_conv_{data_name}.npy")
#     # pred_FNO = np.load(f"{folder_name}pred_FNO_{data_name}.npy")
#     # pred = np.load(f"{folder_name}NODE_pred_{data_name}.npy")
#     correlations = np.zeros(pred.shape[1])
#     correlation_tri = np.zeros(pred.shape[1])
#     correlation_convL = np.zeros(pred.shape[1])
#     correlation_FNO = np.zeros(pred.shape[1])
#     batch =4
#     for t in range (pred.shape[1]):
#         pred_flat = pred[batch,t,0].flatten()
#         ref_flat = data_truth[batch,t,0].flatten()
#         pred_convL_flat = pred_convL[batch,t,0].flatten()
#         pred_tri_flat = pred_tri[batch,t,0].flatten()
#         pred_FNO_flat = pred_FNO[batch,t,0].flatten()
#         corr, _ = pearsonr(ref_flat, pred_flat)
#         corr_convL,_ = pearsonr(ref_flat,pred_convL_flat)
#         corr_tri,_ = pearsonr(ref_flat,pred_tri_flat)
#         corr_FNO,_ = pearsonr(ref_flat,pred_FNO_flat)
#         correlations[t] = corr
#         correlation_tri[t] = corr_tri
#         correlation_convL[t] = corr_convL
#         correlation_FNO[t] = corr_FNO
#     color_profile = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494'] # from light to dark
#     fig,axs = plt.subplots(1,1,figsize=(5,5))
#     axs.set_xticks(np.arange(0,pred.shape[1],1))
#     axs.plot(np.arange(0,pred.shape[1],1),correlations,color=color_profile[-1],label="Ours")
#     axs.plot(np.arange(0,pred.shape[1],1),correlation_convL,color = color_profile[-2],label="ConvLSTM",alpha=0.6)
#     axs.plot(np.arange(0,pred.shape[1],1),correlation_FNO,color = color_profile[-3],label="FNO",alpha=0.6)
#     axs.plot(np.arange(0,pred.shape[1],1),correlation_tri,color = color_profile[-4],label="TriLinear",alpha=0.6)
#     axs.scatter(np.arange(0,pred.shape[1],1),correlations,color=color_profile[-1])
#     axs.scatter(np.arange(0,pred.shape[1],1),correlation_convL,color = color_profile[-2])
#     axs.scatter(np.arange(0,pred.shape[1],1),correlation_FNO,color = color_profile[-3])
#     axs.scatter(np.arange(0,pred.shape[1],1),correlation_tri,color = color_profile[-4])
#     axs.axhline(y = 0.95, color = 'k', linestyle = 'dashed',alpha=0.5,label="95% reference line") 
#     axs.legend(fontsize=12)
#     axs.set_xticks(np.arange(0,pred.shape[1],5),[0,None,0.5,None,1])
#     if data_name =="RBC":
#         axs.set_yticks(np.arange(0.8,1,0.1))
#     else:
#         axs.set_yticks(np.arange(0.75,1,0.1))
#     axs.set_ylabel("Vorticity correlation",fontsize=12)
#     axs.set_xlabel("time",fontsize=12)
#     # axs.set_title(f"vorticity correlation -- {data_name}")
#     # fig.savefig(f"vorticity_correlation_{data_name}.png",dpi=300,bbox_inches='tight')

#     fig.savefig(f"vorticity_correlation_{data_name}.pdf",bbox_inches='tight')

#     return print("voritcity correlation plot done")