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

def eval_ConvLSTM_POD(model_path,in_channels=3):
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
    inputs,target,output = process_loader_baseline_viz(test3_loader, model, device, args)
    return inputs,target,output

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


def eval_NODE_POD(model_path,num_snapshots=20,task_dt=1,result_normalization=False):
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
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size = 32, 
                                                      data_path = args.data_path,
                                                      num_snapshots = 20,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=args.in_channels,)

    inputs,targets,preds =process_loader_NODE_viz(test3_loader, model, device,task_dt=1, n_snapshots=20)
    # Process the last batch outputs (optional, depending on your needs)
    return inputs,targets, preds

if __name__ == "__main__":
    path_lr_sim = {
        "PASR_DT_lrsim_1024_s4_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s4_v0_1537.pt",
        # "PASR_DT_lrsim_1024_s8_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s8_v0_7228.pt",
        # "PASR_DT_lrsim_1024_s16_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s16_v0_5143.pt",
        # "PASR_DT_lrsim_1024_s4_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s4_v0_8137.pt",
        "PASR_DT_lrsim_1024_s8_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s8_v0_9438.pt",
        "PASR_DT_lrsim_1024_s16_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s16_v0_4342.pt",
        # "ConvLSTM_DT_1024_s4_v0":"results/ConvLSTM_DT_1024_s4_v0_sequenceLR3785_checkpoint.pt",
        # "ConvLSTM_DT_1024_s8_v0":"results/ConvLSTM_DT_1024_s8_v0_sequenceLR7399_checkpoint.pt",
        # "ConvLSTM_DT_1024_s16_v0":"results/ConvLSTM_DT_1024_s16_v0_sequenceLR6654_checkpoint.pt",
        # "FNO_DT_1024_s4_v0":"results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt",
        # "FNO_DT_1024_s8_v0":"results/FNO_data_DT_1024_s8_v0_sequenceLR_1306.pt",
        # "FNO_DT_1024_s16_v0":"results/FNO_data_DT_1024_s16_v0_sequenceLR_4373.pt",
        # "TriLinear_DT_1024_s4_v0":"results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt",
        # "TriLinear_DT_1024_s8_v0":"results/FNO_data_DT_1024_s8_v0_sequenceLR_1306.pt",
        # "TriLinear_DT_1024_s16_v0":"results/FNO_data_DT_1024_s16_v0_sequenceLR_4373.pt",
    }
    def get_POD(data,channel=0,number_components=3):
        # data input should be in shape of B,T,H,W
        import numpy as np
        from sklearn.decomposition import PCA
        pca = PCA(n_components=number_components)
        data = data.reshape(data.shape[0]*data.shape[1],-1) # flatten the spatial dimension as features
        pca.fit(data) # using sklearn data shape should be (sample, features) 
        '''
        if use torch.svd, data shape should be (feature, sample)
        
        Note that the data used in POD should be fluctuation only (data - data.mean(dim=time_axis)). 
        
        For this decaying turbulence dataset in particular, the time average is 0. So we didn't normalize it. 

        Details refer to https://arc.aiaa.org/doi/10.2514/1.j056060
        '''
    # for name,model_path in path_lr_sim.items():
    #     inputs,targets,preds = eval_NODE_POD(model_path)
    inputs,targets,preds1 = eval_NODE_POD(path_lr_sim["PASR_DT_lrsim_1024_s4_v0_euler"])
    inputs,targets,preds2 = eval_NODE_POD(path_lr_sim["PASR_DT_lrsim_1024_s8_v0_rk4"])
    inputs,targets,preds3 = eval_NODE_POD(path_lr_sim["PASR_DT_lrsim_1024_s16_v0_rk4"])
    
    pca0 = get_POD(targets[:10,:-1,0,:,:])
    pca1 = get_POD(preds1[:10,:-1,0,:,:])
    pca2 = get_POD(preds2[:10,:-1,0,:,:])
    pca3 = get_POD(preds3[:10,:-1,0,:,:])
    mode0 = []
    mode1 = []
    mode2 = []
    mode3 = []
    import seaborn
    colormap = seaborn.cm.icefire
    fig,ax = plt.subplots(4,3,figsize=(3,4))
    for i in range (3):
        mode0.append(pca0.components_[i].reshape(1024,1024))
        mode1.append(pca1.components_[i].reshape(1024,1024))
        mode2.append(pca2.components_[i].reshape(1024,1024))
        mode3.append(pca3.components_[i].reshape(1024,1024))
        ax[0,i].imshow(mode0[i],cmap=colormap)
        ax[1,i].imshow(mode1[i],cmap=colormap)
        ax[2,i].imshow(mode2[i],cmap=colormap)
        ax[3,i].imshow(mode3[i],cmap=colormap)
    # ax[0,0].set_title("First Mode",fontsize = 11)
    # ax[0,1].set_title("Second Mode",fontsize = 11)
    # ax[0,2].set_title("Third Mode",fontsize = 11)
    for i in range(4):
        for j in range(3):
            ax[i,j].axis("off")
    fig.savefig("PCA_x4x8x16.pdf",bbox_inches="tight",transparent=True)

# for name,model_path in path_lr_sim.items():
#     if name.startswith("PASR"):
#         rfne,mse,mae,inf,ssim,psnr,cumRFNE= eval_NODE(model_path)
#         print(f"RFNE (with-roll-out) {rfne[0][:].mean()}; RFNE(No-roll-out) {rfne[1][:].mean()}; RFNE (single trajectory): {rfne[2][:].mean()},cumulative RFNE {cumRFNE[1][:].mean()}")
#     elif name.startswith("FNO"):
#         rfne,mse,mae,inf,ssim,psnr,cumRFNE= eval_FNO(model_path)
#         print(f"RFNE (with-roll-out) {rfne[0][:].mean()}; RFNE(No-roll-out) {rfne[1][:].mean()}; RFNE (single trajectory): {rfne[2][:].mean()},cumulative RFNE {cumRFNE[1][:].mean()}")
#     elif name.startswith("ConvLSTM"):
#         rfne,mse,mae,inf,ssim,psnr,cumRFNE= eval_ConvLSTM(model_path)
#         print(f"RFNE (with-roll-out) {rfne[0][:].mean()}; RFNE(No-roll-out) {rfne[1][:].mean()}; RFNE (single trajectory): {rfne[2][:].mean()},cumulative RFNE {cumRFNE[1][:].mean()}")
#     else:
#         rfne,mse,mae,inf,ssim,psnr,cumRFNE= eval_TriLinear(model_path)
#         print(f"RFNE (with-roll-out) {rfne[0][:].mean()}; RFNE(No-roll-out) {rfne[1][:].mean()}; RFNE (single trajectory): {rfne[2][:].mean()},cumulative RFNE {cumRFNE[1][:].mean()}")
#     dump_factor = 1
#     dump_json(name,rfne[dump_factor],mae[dump_factor],mse[dump_factor],inf[dump_factor],ssim[dump_factor],psnr[dump_factor],cumRFNE[dump_factor])
    # elif name.startswith("FNO"):
    #     input,target,pred,rfne,mse,mae,inf,ssim,psnr= eval_FNO(model_path)
    # elif name.startswith("ConvLSTM"):
    #     input,target,pred,rfne,mse,mae,inf,ssim,psnr= eval_ConvLSTM(model_path)
    # elif name.startswith("Tri-Linear"):
    #     input,target,pred,rfne,mse,mae,inf,ssim,psnr= eval_TriLinear(model_path)





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
