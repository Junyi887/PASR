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
import matplotlib as mpl
import matplotlib.transforms as transforms
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FixedLocator, FixedFormatter

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


# def plot_energy_specturm_overlay(data_name,folder_name ="4090_results/"):
#     data_truth = np.load(f"{folder_name}hr_target_{data_name}.npy")
#     pred_tri = np.load(f"{folder_name}pred_tri_{data_name}.npy")
#     pred_convL = np.load(f"{folder_name}pred_conv_{data_name}.npy")
#     pred_FNO = np.load(f"{folder_name}pred_FNO_{data_name}.npy")
#     pred = np.load(f"{folder_name}NODE_pred_{data_name}.npy")
#     print(pred.shape)
    
#     for batch in range(12):
#         u_truth,v_truth = data_truth[batch,-5:,1]-data_truth.mean(),data_truth[batch,-5:,2]-data_truth.mean()
#         u_pred,v_pred = pred[batch,-5:,1]-pred.mean(),pred[batch,-5:,2]-pred.mean()
#         u_pred_tri,v_pred_tri = pred_tri[batch,-5:,1]-pred_tri.mean(),pred_tri[batch,-5:,2]-pred_tri.mean()
#         u_pred_convL,v_pred_convL = pred_convL[batch,-5:,1]-pred_convL.mean(),pred_convL[batch,-5:,2]-pred_convL.mean()
#         u_pred_FNO,v_pred_FNO = pred_FNO[batch,-5:,1]-pred_FNO.mean(),pred_FNO[batch,-5:,2]-pred_FNO.mean()
#         realsize_truth, EK_avsphr_truth,result_dict_truth = energy_specturm(u_truth,v_truth)
#         realsize_pred, EK_avsphr_pred,result_dict_pred = energy_specturm(u_pred,v_pred)
#         realsize_pred_tri, EK_avsphr_pred_tri,result_dict_pred_tri = energy_specturm(u_pred_tri,v_pred_tri)
#         realsize_pred_convL, EK_avsphr_pred_convL,result_dict_pred_convL = energy_specturm(u_pred_convL,v_pred_convL)
#         realsize_pred_FNO, EK_avsphr_pred_FNO,result_dict_pred_FNO = energy_specturm(u_pred_FNO,v_pred_FNO)
#         if data_name == "DT":
#             x_bound = 25
#             zoom_in_localtion = [0.2, 0.2, 0.4, 0.2]
#             x1, x2, y1, y2 = 11, 12, 6e-5, 6.5e-4  # subregion of the original image
#             save_batch = 3
#             xticks = [8,10]
#             y_bound = [1e-9,0.99999]
#             xticks_label = [r'$6 \times 10^0$' ,r'$10^1$']
#         if data_name =="RBC":
#             x_bound = 25
#             zoom_in_localtion = [0.13, 0.25, 0.49, 0.2]
#             x1, x2, y1, y2 = 9, 11, 5e-5, 1e-4  # subregion of the original image
#             save_batch = 9
#             y_bound = [1e-9,1e-1]
#             xticks = [9,11]
#             xticks_label = [r'$9 \times 10^0$' ,r'$10^1$']
#         fig, ax = plt.subplots(figsize=(3.3,3.6))
#         # ax.set_title(f"Kinetic Energy Spectrum -- {data_name}")
#         ax.set_xlabel(r"k",fontsize=11)
#         ax.set_ylabel(r"TKE of the k$^{th}$ wavenumber",fontsize=11)
#         # Use ax.set_xlim to avoid issues with plt when having multiple axes
#         # Replace 0 with a small positive number (e.g., 1)
#         ax.set_xlim(left=1, right=x_bound)
#         ax.set_ylim(bottom=y_bound[0], top=y_bound[1])
#         ax.set_xscale('log')
#         ax.set_yscale('log')
#         import seaborn as sns
#         color_platte = sns.color_palette("YlGnBu_r", 5)
#         alpha = 0.8
#         ax.loglog(np.arange(0, realsize_truth), EK_avsphr_truth[0:realsize_truth], c = 'k',label="Truth",alpha=1)
#         ax.loglog(np.arange(0, realsize_pred), EK_avsphr_pred[0:realsize_pred], c=color_platte[0], alpha=alpha,label="Ours")
#         ax.loglog(np.arange(0, realsize_pred_convL), EK_avsphr_pred_convL[0:realsize_pred_convL], c=color_platte[1], alpha=alpha,label="ConvLSTM")
#         ax.loglog(np.arange(0, realsize_pred_tri), EK_avsphr_pred_tri[0:realsize_pred_tri], c = color_platte[2], alpha=alpha,label="Trilinear")
#         ax.loglog(np.arange(0, realsize_pred_FNO), EK_avsphr_pred_FNO[0:realsize_pred_FNO], c=color_platte[3], alpha=alpha,label="FNO3D")

#         axins = ax.inset_axes(zoom_in_localtion, xlim=(x1, x2), ylim=(y1, y2)) # [x0, y0, width, height]

#         # # Plot on the inset
#         axins.plot(np.arange(0, realsize_truth), EK_avsphr_truth[0:realsize_truth], c='k', alpha=1)
#         axins.plot(np.arange(0, realsize_pred), EK_avsphr_pred[0:realsize_pred],  c=color_platte[0], alpha=alpha)
#         axins.plot(np.arange(0, realsize_pred_convL), EK_avsphr_pred_convL[0:realsize_pred_convL],  c=color_platte[1], alpha=alpha)
#         axins.plot(np.arange(0, realsize_pred_FNO), EK_avsphr_pred_FNO[0:realsize_pred_FNO], c=color_platte[2], alpha=alpha)
#         axins.plot(np.arange(0, realsize_pred_tri), EK_avsphr_pred_tri[0:realsize_pred_tri], c=color_platte[3], alpha=alpha)


#         axins.set_yscale('log')
#         axins.set_xticks(xticks,xticks_label)
#         # axins.set_xscale('log')
#         ax.indicate_inset_zoom(axins, edgecolor="black")
#         ax.legend(loc='upper right',fontsize=9)
#         # Save the figure using fig.savefig instead of plt.savefig to avoid context issues
#         fig.savefig(f"{data_name}_energy_specturm_{batch}.png", dpi =300,bbox_inches='tight')
#         if batch ==save_batch:
#             fig.savefig(f"PaperWrite/{data_name}_energy_specturm_{batch}.pdf", bbox_inches='tight')
#         plt.close(fig)  # Close the figure to free memory

#     return print("energy specturm plot done")


def plot_vorticity_correlation(data_name,folder_name ="4090_results/"):
    data_truth = np.load(f"{folder_name}hr_target_{data_name}.npy")
    pred_tri = np.load(f"{folder_name}pred_tri_{data_name}.npy")
    pred_convL = np.load(f"{folder_name}pred_conv_{data_name}.npy")
    pred_FNO = np.load(f"{folder_name}pred_FNO_{data_name}.npy")
    pred = np.load(f"{folder_name}NODE_pred_{data_name}.npy")
    correlations = np.zeros(pred.shape[1])
    correlation_tri = np.zeros(pred.shape[1])
    correlation_convL = np.zeros(pred.shape[1])
    correlation_FNO = np.zeros(pred.shape[1])
    batch =4
    for t in range (pred.shape[1]):
        pred_flat = pred[batch,t,0].flatten()
        ref_flat = data_truth[batch,t,0].flatten()
        pred_convL_flat = pred_convL[batch,t,0].flatten()
        pred_tri_flat = pred_tri[batch,t,0].flatten()
        pred_FNO_flat = pred_FNO[batch,t,0].flatten()
        corr, _ = pearsonr(ref_flat, pred_flat)
        corr_convL,_ = pearsonr(ref_flat,pred_convL_flat)
        corr_tri,_ = pearsonr(ref_flat,pred_tri_flat)
        corr_FNO,_ = pearsonr(ref_flat,pred_FNO_flat)
        correlations[t] = corr
        correlation_tri[t] = corr_tri
        correlation_convL[t] = corr_convL
        correlation_FNO[t] = corr_FNO
    # color_profile = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494'] # from light to dark
    color_profile = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    fig,axs = plt.subplots(1,1,figsize=(3.6,3.6*6/10))
    alpha = 1
    axs.set_xticks(np.arange(0,pred.shape[1],1))
    axs.plot(np.arange(0,pred.shape[1],1),correlations,color=color_profile[0],label="Ours")
    axs.plot(np.arange(0,pred.shape[1],1),correlation_convL,color = color_profile[1],label="ConvLSTM",alpha=alpha)
    axs.plot(np.arange(0,pred.shape[1],1),correlation_FNO,color = color_profile[2],label="FNO",alpha=alpha)
    axs.plot(np.arange(0,pred.shape[1],1),correlation_tri,color = color_profile[3],label="TriLinear",alpha=alpha)
    axs.scatter(np.arange(0,pred.shape[1],1),correlations,color=color_profile[0],marker =".")
    axs.scatter(np.arange(0,pred.shape[1],1),correlation_convL,color = color_profile[1],marker ="^")
    axs.scatter(np.arange(0,pred.shape[1],1),correlation_FNO,color = color_profile[2],marker= "d")
    axs.scatter(np.arange(0,pred.shape[1],1),correlation_tri,color = color_profile[3],marker = "*")
    axs.axhline(y = 0.95, color = 'k', linestyle = 'dashed',alpha=0.5,label="95% reference line") 
    if data_name =="RBC":
        axs.legend(fontsize=9,frameon=False) 
    axs.set_xticks(np.arange(0,pred.shape[1],5),[0,None,0.5,None,1])
    axs.set_yticks(np.arange(0.85,1,0.1))
    axs.set_ylabel("Vorticity correlation",fontsize=9)
    axs.set_xlabel("time",fontsize=9)
    axs.set_ylim(0.85,1.01)
    fig.tight_layout()
    # axs.set_title(f"vorticity correlation -- {data_name}")
    # fig.savefig(f"vorticity_correlation_{data_name}.png",dpi=300,bbox_inches='tight')
    np.save(f"vorticity_correlation_{data_name}.npy",correlations)
    np.save(f"vorticity_correlation_tri_{data_name}.npy",correlation_tri)
    np.save(f"vorticity_correlation_convL_{data_name}.npy",correlation_convL)
    np.save(f"vorticity_correlation_FNO_{data_name}.npy",correlation_FNO)

    fig.savefig(f"vorticity_correlation_{data_name}.pdf",bbox_inches='tight',transparent=True)

    return print("voritcity correlation plot done")

def plot_DT_comparision():
    hr_target = np.load("hr_target_DT.npy")
    pred_tri = np.load("pred_tri_DT.npy")
    pred_conv = np.load("pred_conv_DT.npy")
    pred_FNO = np.load("pred_FNO_DT.npy")
    pred_NODE = np.load("pred_NODE_DT.npy")
    fig,ax = plt.subplots(3,5,figsize=(28,18))
    import seaborn
    for batch in [0,1,2,5,8]:
        for i in range(3):
            ax[i,0].imshow(hr_target[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
            ax[i,1].imshow(pred_tri[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
            ax[i,2].imshow(pred_FNO[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
            ax[i,3].imshow(pred_conv[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
            ax[i,4].imshow(pred_NODE[batch,i*10,0],vmin=hr_target.min(),vmax=hr_target.max(),cmap = seaborn.cm.icefire)
            ax[i,0].set_axis_off()
            ax[i,1].set_axis_off()
            ax[i,2].set_axis_off()
            ax[i,3].set_axis_off()
            ax[i,4].set_axis_off()
        fig.savefig(f"Decay_turb_baseline_{batch}.png",bbox_inches='tight')
    return print("plot done")

def plot_RBC_comparision():
    hr_target = np.load("hr_target_RBC.npy")
    pred_tri = np.load("pred_tri_RBC.npy")
    pred_conv = np.load("pred_conv_RBC.npy")
    pred_FNO = np.load("pred_FNO_RBC.npy")
    pred_NODE = np.load("pred_NODE_RBC.npy")
    fig,ax = plt.subplots(5,3,figsize=(20,8))

    import seaborn
    for batch in [0,1,2,3,4]:
        for i in range(3):
            vmin = hr_target[batch,...,5:-5,5:-5].min()
            vmax = hr_target[batch,...,5:-5,5:-5].max()
            ax[0,i].imshow(hr_target[batch,i*10,0,5:-5,5:-5].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.icefire)
            ax[0,i].set_axis_off()
            ax[1,i].imshow(pred_tri[batch,i*10,0,5:-5,5:-5].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.icefire)
            ax[1,i].set_axis_off()
            ax[2,i].imshow(pred_FNO[batch,i*10,0,5:-5,5:-5].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.icefire)
            ax[2,i].set_axis_off()
            ax[3,i].imshow(pred_conv[batch,i*10,0,5:-5,5:-5].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.icefire)
            ax[3,i].set_axis_off()
            ax[4,i].imshow(pred_NODE[batch,i*10,0,5:-5,5:-5].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.icefire)
            ax[4,i].set_axis_off()
        fig.savefig(f"RBC_baseline_{batch}.png",bbox_inches='tight')

def plot_climate_comparision():
    hr_target = np.load("hr_target_climate_s4_sig1.npy")
    pred_tri = np.load("pred_climate_s4_sig1_trilinear.npy")
    pred_conv = np.load("pred_climate_s4_sig1_ConvLSTM.npy")
    pred_FNO = np.load("pred_climate_s4_sig1_FNO.npy")
    # pred_NODE = np.load("pred_NODE_climate.npy")
    fig,ax = plt.subplots(5,3,figsize=(10,8))
    print(hr_target.shape)
    import seaborn
    import cmocean
    for batch in [0,1,2,3,4]:
        for i in range(3):
            vmin = hr_target.min()
            vmax = hr_target.max()
            ax[0,i].imshow(hr_target[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = cmocean.cm.balance)
            ax[0,i].set_axis_off()
            ax[1,i].imshow(pred_tri[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = cmocean.cm.balance)
            ax[1,i].set_axis_off()
            ax[2,i].imshow(pred_FNO[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = cmocean.cm.balance)
            ax[2,i].set_axis_off()
            ax[3,i].imshow(pred_conv[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = cmocean.cm.balance)
            ax[3,i].set_axis_off()
            # ax[4,i].imshow(pred_NODE[batch,i*10,0,:,:].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.balance)
            # ax[4,i].set_axis_off()
        fig.savefig(f"Climate_baseline_{batch}.png",bbox_inches='tight')

def plot_climate_comparision_normalized():
    hr_target = np.load("hr_target_climate_s4_sig1.npy")
    pred_tri = np.load("pred_climate_s4_sig1_trilinear.npy")
    pred_conv = np.load("pred_climate_s4_sig1_ConvLSTM.npy")
    pred_FNO = np.load("pred_climate_s4_sig1_FNO.npy")
    # pred_NODE = np.load("pred_NODE_climate.npy")
    fig,ax = plt.subplots(5,3,figsize=(10,8))
    print(hr_target.shape)
    import seaborn
    import cmocean
    hr_target = (hr_target-hr_target.mean())/(hr_target.std())
    pred_tri = (pred_tri-pred_tri.mean())/(pred_tri.std())
    pred_conv = (pred_conv-pred_conv.mean())/(pred_conv.std())
    pred_FNO = (pred_FNO-pred_FNO.mean())/(pred_FNO.std())
    

    for batch in [0,1,2,3,4]:
        for i in range(3):
            vmin = hr_target.min()
            vmax = hr_target.max()
            ax[0,i].imshow(hr_target[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = cmocean.cm.balance)
            ax[0,i].set_axis_off()
            ax[1,i].imshow(pred_tri[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = cmocean.cm.balance)
            ax[1,i].set_axis_off()
            ax[2,i].imshow(pred_FNO[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = cmocean.cm.balance)
            ax[2,i].set_axis_off()
            ax[3,i].imshow(pred_conv[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = cmocean.cm.balance)
            ax[3,i].set_axis_off()
            # ax[4,i].imshow(pred_NODE[batch,i*10,0,:,:].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.balance)
            # ax[4,i].set_axis_off()
        fig.savefig(f"n_Climate_baseline_{batch}.png",bbox_inches='tight')

def plot_vorticity_correlation_extrapolation(data_name,folder_name ="4090_results/"):
    # data_truth = np.load(f"Extrapolation_hr_target_80_{data_name}.npy")
    # pred_rk4 = np.load(f"Extrapolation_pred_{data_name}_rk4_NODE.npy")
    # pred_euler = np.load(f"Extrapolation_pred_{data_name}_euler_NODE.npy")
    data_truth = np.load(f"{folder_name}Extrapolation_loop_back_hr_target_80_{data_name}.npy")
    pred_rk4 = np.load(f"{folder_name}Extrapolation_loop_back_pred_{data_name}_rk4_NODE.npy")
    pred_euler = np.load(f"{folder_name}Extrapolation_loop_back_pred_{data_name}_euler_NODE.npy")
    print(pred_rk4.shape)
    print(data_truth.shape)
    print(pred_euler.shape)
    correlations = np.zeros(pred_rk4.shape[1])
    correlation_euler = np.zeros(pred_rk4.shape[1])
    correlation_rk4 = np.zeros(pred_rk4.shape[1])
    batch =4
    for batch in range(5):
        for t in range (pred_rk4.shape[1]):
            ref_flat = data_truth[batch,t,0].flatten()
            pred_rk4_flat = pred_rk4[batch,t,0].flatten()
            pred_euler_flat = pred_euler[batch,t,0].flatten()
            corr_rk4,_ = pearsonr(ref_flat,pred_rk4_flat)
            corr_euler,_ = pearsonr(ref_flat,pred_euler_flat)
            correlation_euler[t] = corr_euler
            correlation_rk4[t] = corr_rk4

        fig,axs = plt.subplots(1,1,figsize=(3,3))
        axs.plot(np.arange(0,pred_rk4.shape[1],1),correlation_rk4,label="RK4")
        axs.plot(np.arange(0,pred_rk4.shape[1],1),correlation_euler,label="Euler")
        axs.legend(fontsize=12)
        axs.set_ylabel("vorticity correlation",fontsize=12)
        axs.set_xlabel("time",fontsize=12)
        # axs.set_title(f"{data_name}")
        axs.set_xticks(np.arange(0,pred_rk4.shape[1],20),[0,1,2,3,4])
        axs.set_yticks(np.arange(0.8,1.05,0.1),[0.8,0.9,1])
        axs.set_ylim(0.8,1.01) if data_name =="decay_turb" else axs.set_ylim(0.9,1.01)
        fig.savefig(f"vorticity_correlation_{data_name}_{batch}_loop.pdf",bbox_inches='tight')
    return print("voritcity correlation extrapolation loop back plot done")



def plot_energy_specturm_overlay2(data_name,folder_name ="4090_results/"):
    data_truth = np.load(f"{folder_name}hr_target_{data_name}.npy")
    pred_tri = np.load(f"{folder_name}pred_tri_{data_name}.npy")
    pred_convL = np.load(f"{folder_name}pred_conv_{data_name}.npy")
    pred_FNO = np.load(f"{folder_name}pred_FNO_{data_name}.npy")
    pred = np.load(f"{folder_name}NODE_pred_{data_name}.npy")
    print(pred.shape)
    if data_name == "DT":
        x_bound = 25
        zoom_in_localtion = [0.2, 0.2, 0.4, 0.2]
        x1, x2, y1, y2 = 4.9, 6.8, 1e-3, 2e-3  # subregion of the original image
        save_batch = 3
        xticks = [8,10]
        yticks = [1e-5,1e-4]
        yticks_label = [r'$10^{-5}$' ,r'10^{-4}$']
        y_bound = [1e-6,1e-1]
        xticks_label = [r'$6 \times 10^0$' ,r'$10^1$']
        batch_start = 15
        batch_end = 21
        zoom_in_factor = 6
    if data_name =="RBC":
        x_bound = 25
        zoom_in_localtion = [0.17, 0.29, 0.49, 0.2]
        x1, x2, y1, y2 = 7.7, 9.3, 1e-4, 3e-4   # subregion of the original image 
        save_batch = 9
        y_bound = [1e-6,1e-1]
        # xticks = [x1,x2]
        # xticks_label = [r'$9 \times 10^0$' ,r'$10^1$']
        # yticks = [y1,y2]
        # yticks_label = [r'$10^{-5}$' ,r'$10^{-4}$']
        batch_start = 0
        batch_end = 20
        zoom_in_factor = 6 
    for batch in range(12):
        u_truth,v_truth = data_truth[batch,batch_start:batch_end,1]-data_truth.mean(),data_truth[batch,batch_start:batch_end,2]-data_truth.mean()
        u_pred,v_pred = pred[batch,batch_start:batch_end,1]-pred.mean(),pred[batch,batch_start:batch_end,2]-pred.mean()
        u_pred_tri,v_pred_tri = pred_tri[batch,batch_start:batch_end,1]-pred_tri.mean(),pred_tri[batch,batch_start:batch_end,2]-pred_tri.mean()
        u_pred_convL,v_pred_convL = pred_convL[batch,batch_start:batch_end,1]-pred_convL.mean(),pred_convL[batch,batch_start:batch_end,2]-pred_convL.mean()
        u_pred_FNO,v_pred_FNO = pred_FNO[batch,batch_start:batch_end,1]-pred_FNO.mean(),pred_FNO[batch,batch_start:batch_end,2]-pred_FNO.mean()
        realsize_truth, EK_avsphr_truth,result_dict_truth = energy_specturm(u_truth,v_truth)
        realsize_pred, EK_avsphr_pred,result_dict_pred = energy_specturm(u_pred,v_pred)
        realsize_pred_tri, EK_avsphr_pred_tri,result_dict_pred_tri = energy_specturm(u_pred_tri,v_pred_tri)
        realsize_pred_convL, EK_avsphr_pred_convL,result_dict_pred_convL = energy_specturm(u_pred_convL,v_pred_convL)
        realsize_pred_FNO, EK_avsphr_pred_FNO,result_dict_pred_FNO = energy_specturm(u_pred_FNO,v_pred_FNO)

        fig, ax = plt.subplots(figsize=(3.6,3.6*6/10))
        # ax.set_title(f"Kinetic Energy Spectrum -- {data_name}")
        ax.set_xlabel(r"Wavenumber k",fontsize=10)
        ax.set_ylabel(r"Energy E[k]",fontsize=10)
        # Use ax.set_xlim to avoid issues with plt when having multiple axes
        # Replace 0 with a small positive number (e.g., 1)
        ax.set_xlim(left=1, right=x_bound)
        ax.set_ylim(bottom=y_bound[0], top=y_bound[1])
        ax.set_xscale('log')
        ax.set_yscale('log')
        import seaborn as sns
        color_platte = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
        # color_platte = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e'] # first on color brewer
        # color_platte = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854'] # second 

        # Create a custom color map from the listed colors
        alpha = 1.0
        ax.loglog(np.arange(0, realsize_truth), EK_avsphr_truth[0:realsize_truth], c = 'k',label="Truth",alpha=1)
        ax.loglog(np.arange(0, realsize_pred), EK_avsphr_pred[0:realsize_pred], c=color_platte[0], alpha=alpha,label="Ours")
        ax.loglog(np.arange(0, realsize_pred_convL), EK_avsphr_pred_convL[0:realsize_pred_convL], c=color_platte[1], alpha=alpha,label="ConvLSTM")
        ax.loglog(np.arange(0, realsize_pred_tri), EK_avsphr_pred_tri[0:realsize_pred_tri], c = color_platte[2], alpha=alpha,label="Trilinear")
        ax.loglog(np.arange(0, realsize_pred_FNO), EK_avsphr_pred_FNO[0:realsize_pred_FNO], c=color_platte[3], alpha=alpha,label="FNO3D")
        if batch == save_batch:
            np.save(f"energy_spectrum_realsize_truth_{data_name}.npy",realsize_truth)
            np.save(f"energy_spectrum_realsize_pred_{data_name}.npy",realsize_pred)
            np.save(f"energy_spectrum_realsize_pred_tri_{data_name}.npy",realsize_pred_tri)
            np.save(f"energy_spectrum_realsize_pred_convL_{data_name}.npy",realsize_pred_convL)
            np.save(f"energy_spectrum_realsize_pred_FNO_{data_name}.npy",realsize_pred_FNO)
            np.save(f"energy_spectrum_EK_avsphr_truth_{data_name}.npy",EK_avsphr_truth)
            np.save(f"energy_spectrum_EK_avsphr_pred_{data_name}.npy",EK_avsphr_pred)
            np.save(f"energy_spectrum_EK_avsphr_pred_tri_{data_name}.npy",EK_avsphr_pred_tri)
            np.save(f"energy_spectrum_EK_avsphr_pred_convL_{data_name}.npy",EK_avsphr_pred_convL)
            np.save(f"energy_spectrum_EK_avsphr_pred_FNO_{data_name}.npy",EK_avsphr_pred_FNO)

        axins = zoomed_inset_axes(ax,zoom_in_factor,loc='lower left') # [x0, y0, width, height]
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        # # Plot on the inset
        axins.loglog(np.arange(0, realsize_truth), EK_avsphr_truth[0:realsize_truth], c='k', alpha=1)
        axins.loglog(np.arange(0, realsize_pred), EK_avsphr_pred[0:realsize_pred],  c=color_platte[0], alpha=alpha)
        axins.loglog(np.arange(0, realsize_pred_convL), EK_avsphr_pred_convL[0:realsize_pred_convL],  c=color_platte[1], alpha=alpha)
        axins.loglog(np.arange(0, realsize_pred_tri), EK_avsphr_pred_tri[0:realsize_pred_tri], c=color_platte[2], alpha=alpha)
        axins.loglog(np.arange(0, realsize_pred_FNO), EK_avsphr_pred_FNO[0:realsize_pred_FNO], c=color_platte[3], alpha=alpha)

        _patch, pp1, pp2 = mark_inset(ax, axins, loc1=2, loc2=4, fc='none', ec='0.3', lw=1.0, color='k') 
        # axins.set_yscale('log')
        # axins.set_xscale('log')

        # axins.set_xticks(xticks,['',''])
        # axins.set_yticks(yticks,['',''])
        axins.xaxis.set_tick_params(labelbottom=False)
        axins.yaxis.set_tick_params(labelleft=False)
        axins.set_xticks([])
        axins.set_yticks([])
        axins.minorticks_off()
        fig.tight_layout()
        plt.draw()
        # ax.indicate_inset_zoom(axins, edgecolor="black")
        # if data_name =="RBC":
        #     ax.legend(loc='upper right',fontsize=8.5,frameon=False) 
        # Save the figure using fig.savefig instead of plt.savefig to avoid context issues
        fig.savefig(f"{data_name}_energy_specturm_{batch}.png", dpi =300,bbox_inches='tight')
        if batch ==save_batch:
            fig.savefig(f"PaperWrite/{data_name}_energy_specturm_{batch}_v2.pdf", bbox_inches='tight',transparent=True)
        plt.close(fig)  # Close the figure to free memory

    return print("energy specturm plot done")


import numpy as np 
import matplotlib.pyplot as plt
plot_energy_specturm_overlay2("DT")
# plot_RBC_comparision()
plot_energy_specturm_overlay2("RBC")
plot_vorticity_correlation("DT")
plot_vorticity_correlation("RBC")
# plot_climate_comparision()
# plot_climate_comparision_normalized()
# plot_vorticity_correlation_extrapolation("decay_turb")
# plot_vorticity_correlation_extrapolation("rbc")