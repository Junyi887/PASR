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
def eval_FNO_POD(model_path,in_channels=3,batch_size=4):

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
    for i,(inputs,targets) in enumerate(test3_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
    return inputs.cpu().detach().numpy(),targets.cpu().detach().numpy(),output.cpu().detach().numpy()

def eval_ConvLSTM_POD(model_path,in_channels=3):
    checkpoint = torch.load(model_path)
    if "config" in checkpoint.keys():
        print("config")
        config = checkpoint['config']
    else:
        checkpoint1024 = torch.load("results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt")
        checkpoint_data = checkpoint1024
        config = checkpoint_data['config']
        print("load FNO config")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import argparse
    model_state = checkpoint['model_state_dict']
    args = argparse.Namespace()
    args.__dict__.update(config)
    num_snapshots = 20
    # stats_loader = DataInfoLoader(args.data_path+"/*/*.h5",args.data) 
    stats_loader = DataInfoLoader("/pscratch/sd/j/junyi012/DT_lrsim_256_s4_v0/*/*.h5",args.data) #/pscratch/sd/j/junyi012/DT_lrsim_1024_s4_v0
    mean,std = get_normalizer(args,stats_loader)
    img_x, img_y = stats_loader.get_shape()
    steps = num_snapshots + 1 
    effective_step = list(range(0, steps))
    model = PhySR(
        n_feats=32,
        n_layers=[1, 2],  # [n_convlstm, n_resblock]
        upscale_factor=[num_snapshots, args.scale_factor],  # [t_up, s_up]
        shift_mean_paras=[mean, std],  
        step=steps,
        in_channels=in_channels,
        effective_step=effective_step
    )
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    test1_loader,test2_loader,test3_loader = getData_test(upscale_factor = args.scale_factor, 
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size =64, 
                                                      data_path = args.data_path,
                                                      num_snapshots = 20,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=3,)
    for i,(inputs,targets) in enumerate(test3_loader):
        print(i)
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
    return inputs.cpu().detach().numpy(),targets.cpu().detach().numpy(),output.cpu().detach().numpy()

def eval_TriLinear_POD(model_path,in_channels=3,batch_size=16,n_snapshots=20):
    checkpoint = torch.load(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    import argparse
    config = checkpoint['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
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
    for i,(inputs,targets) in enumerate(test3_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
    return inputs.cpu().detach().numpy(),targets.cpu().detach().numpy(),output.cpu().detach().numpy()


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
    # path_lr_sim = {
    #     "PASR_DT_LR_SIM_1024_s4":"results/PASR_ODE_small_data_DT_lrsim_1024_s4_v0_8137.pt",
    #     "PASR_DT_LR_SIM_512_s4":"results/PASR_ODE_small_data_DT_lrsim_512_s4_v0_5019.pt",
    #     "PASR_DT_LR_SIM_256_s4":"results/PASR_ODE_small_data_DT_lrsim_256_s4_v0_2557.pt",
    #     # "ConvLSTM_DT_LR_SIM_1024_s4":"results/ConvLSTM_DT_1024_s4_v0_sequenceLR5733_checkpoint.pt",
    #     # "ConvLSTM_DT_LR_SIM_512_s4":"ConvLSTM_DT_512_s4_v0_sequenceLR3895_checkpoint.pt",
    #     # "ConvLSTM_DT_LR_SIM_256_s4":"ConvLSTM_DT_256_s4_v0_sequenceLR865_checkpoint.pt",
    #     # "FNO_DT_LR_SIM_1024_s4":"results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt",
    #     # "FNO_DT_LR_SIM_512_s4":"results/FNO_data_DT_512_s4_v0_sequenceLR_1.pt",
    #     # "FNO_DT_LR_SIM_256_s4":"results/FNO_data_DT_256_s4_v0_sequenceLR_2508.pt",
    #     # "TriLinear_DT_LR_SIM_512_s4":"results/FNO_data_DT_512_s4_v0_sequenceLR_1.pt",
    #     # "TriLinear_DT_LR_SIM_1024_s4":"results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt",
    #     # "TriLinear_DT_LR_SIM_256_s4":"results/FNO_data_DT_256_s4_v0_sequenceLR_2508.pt",
    # }
    path_lr_sim = {
        #"PASR_DT_lrsim_1024_s4_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s4_v0_1537.pt",
        # "PASR_DT_lrsim_1024_s8_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s8_v0_7228.pt",
        # "PASR_DT_lrsim_1024_s16_v0_euler":"results/PASR_ODE_small_data_DT_lrsim_1024_s16_v0_5143.pt",
        "PASR_DT_lrsim_1024_s4_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s4_v0_8137.pt",
        "PASR_DT_lrsim_1024_s8_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s8_v0_9438.pt",
        "PASR_DT_lrsim_1024_s16_v0_rk4":"results/PASR_ODE_small_data_DT_lrsim_1024_s16_v0_4342.pt",
        "ConvLSTM_DT_1024_s4_v0":"results/ConvLSTM_DT_1024_s4_v0_sequenceLR3785_checkpoint.pt",
        "ConvLSTM_DT_1024_s8_v0":"results/ConvLSTM_DT_1024_s8_v0_sequenceLR7399_checkpoint.pt",
        "ConvLSTM_DT_1024_s16_v0":"results/ConvLSTM_DT_1024_s16_v0_sequenceLR6654_checkpoint.pt",
        "FNO_DT_1024_s4_v0":"results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt",
        "FNO_DT_1024_s8_v0":"results/FNO_data_DT_1024_s8_v0_sequenceLR_1306.pt",
        "FNO_DT_1024_s16_v0":"results/FNO_data_DT_1024_s16_v0_sequenceLR_4373.pt",
        "TriLinear_DT_1024_s4_v0":"results/FNO_data_DT_1024_s4_v0_sequenceLR_6804.pt",
        "TriLinear_DT_1024_s8_v0":"results/FNO_data_DT_1024_s8_v0_sequenceLR_1306.pt",
        "TriLinear_DT_1024_s16_v0":"results/FNO_data_DT_1024_s16_v0_sequenceLR_4373.pt",
    }
    import os
    def get_POD(data,channel=0,number_components=3):
        # data input should be in shape of B,T,H,W
        import numpy as np
        from sklearn.decomposition import PCA
        pca = PCA(n_components=number_components)
        data = data.reshape(data.shape[0]*data.shape[1],-1) # flatten the spatial dimension
        pca.fit(data)
        return pca
    for scale in [4,8,16]:
        if os.path.exists(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_0.npy") == False:
            inputs,targets,preds = eval_NODE_POD(path_lr_sim[f"PASR_DT_lrsim_1024_s{scale}_v0_rk4"])
            _,_,pred_convLSTM = eval_ConvLSTM_POD(path_lr_sim[f"ConvLSTM_DT_1024_s{scale}_v0"])
            _,_,pred_FNO = eval_FNO_POD(path_lr_sim[f"FNO_DT_1024_s{scale}_v0"])
            _,_,pred_Tilinear= eval_TriLinear_POD(path_lr_sim[f"TriLinear_DT_1024_s{scale}_v0"])
            pca0 = get_POD(inputs[:7,0:1,:,:])
            pca1 = get_POD(targets[:7,:-1,0,:,:])
            pca2 = get_POD(preds[:7,:-1,0,:,:])
            pca3 = get_POD(pred_convLSTM[:7,0,:-1,:,:])
            pca4 = get_POD(pred_FNO[:7,0,:-1,:,:])
            pca5 = get_POD(pred_Tilinear[:7,0,:-1,:,:])
            np.save(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_0.npy",pca0.components_)
            np.save(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_1.npy",pca1.components_)
            np.save(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_2.npy",pca2.components_)
            np.save(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_3.npy",pca3.components_)
            np.save(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_4.npy",pca4.components_)
            np.save(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_5.npy",pca5.components_)
            pca_components0 = pca0.components_
            pca_components1 = pca1.components_
            pca_components2 = pca2.components_
            pca_components3 = pca3.components_
            pca_components4 = pca4.components_
            pca_components5 = pca5.components_
        else:
            pca_components0 = np.load(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_0.npy")
            pca_components1 = np.load(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_1.npy")
            pca_components2 = np.load(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_2.npy")
            pca_components3 = np.load(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_3.npy")
            pca_components4 = np.load(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_4.npy")
            pca_components5 = np.load(f"/pscratch/sd/j/junyi012/PASR_v0/results_buffer/PAC_x{scale}_5.npy")
        
        mode0 =[]
        mode1 = []
        mode2 = []
        mode3 = []
        mode4 = []
        mode5 = []
        import seaborn
        colormap = seaborn.cm.icefire
        fig,ax = plt.subplots(3,6,figsize=(6.1,3.2))
        for i in range (3):
            # mode1.append(pca1.components_[i].reshape(1024,1024))
            # mode2.append(pca2.components_[i].reshape(1024,1024))
            # mode3.append(pca3.components_[i].reshape(1024,1024))
            # mode4.append(pca4.components_[i].reshape(1024,1024))
            # mode5.append(pca5.components_[i].reshape(1024,1024))
            # mode0.append(pca0.components_[i].reshape(1024//scale,1024//scale))
            mode1.append(pca_components1[i].reshape(1024,1024))
            mode2.append(pca_components2[i].reshape(1024,1024))
            mode3.append(pca_components3[i].reshape(1024,1024))
            mode4.append(pca_components4[i].reshape(1024,1024))
            mode5.append(pca_components5[i].reshape(1024,1024))
            mode0.append(pca_components0[i].reshape(1024//scale,1024//scale))
            ax[i,0].imshow(mode1[i],cmap=colormap)
            ax[i,1].imshow(mode2[i],cmap=colormap)
            ax[i,2].imshow(mode3[i],cmap=colormap)
            ax[i,3].imshow(mode4[i],cmap=colormap)
            ax[i,4].imshow(mode5[i],cmap=colormap)
            ax[i,5].imshow(mode0[i],cmap=colormap)
        for i in range(3):
            for j in range(6):
                ax[i,j].axis("off")
        ax[0,0].set_title("HR",fontsize=9)
        ax[0,1].set_title("Ours",fontsize=9)
        ax[0,2].set_title("ConvLSTM",fontsize=9)
        ax[0,3].set_title("FNO",fontsize=9)
        ax[0,4].set_title("TriLinear",fontsize=9)
        ax[0,5].set_title("LR",fontsize=9)
        fig.tight_layout(w_pad=0.25,h_pad=0.25,pad=0.25)
        # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.2, hspace=0.2)

        fig.savefig(f"PCA_x{scale}.pdf",bbox_inches="tight",transparent=True)

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