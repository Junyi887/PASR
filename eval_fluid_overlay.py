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

def plot_energy_specturm_overlay(data_name):
    data_truth = np.load(f"hr_target_{data_name}.npy")
    pred_tri = np.load(f"pred_tri_{data_name}.npy")
    pred_convL = np.load(f"pred_conv_{data_name}.npy")
    pred_FNO = np.load(f"pred_FNO_{data_name}.npy")
    pred = np.load(f"NODE_pred_{data_name}.npy")
    print(pred.shape)
    for batch in range(12):
        u_truth,v_truth = data_truth[batch,-5:,1]-data_truth.mean(),data_truth[batch,-5:,2]-data_truth.mean()
        u_pred,v_pred = pred[batch,-5:,1]-pred.mean(),pred[batch,-5:,2]-pred.mean()
        u_pred_tri,v_pred_tri = pred_tri[batch,-5:,1]-pred_tri.mean(),pred_tri[batch,-5:,2]-pred_tri.mean()
        u_pred_convL,v_pred_convL = pred_convL[batch,-5:,1]-pred_convL.mean(),pred_convL[batch,-5:,2]-pred_convL.mean()
        u_pred_FNO,v_pred_FNO = pred_FNO[batch,-5:,1]-pred_FNO.mean(),pred_FNO[batch,-5:,2]-pred_FNO.mean()

        # u_pred,v_pred = pred[batch,:,1],pred[batch,:,2]
        # u_pred_tri,v_pred_tri = pred_tri[batch,:,1],pred_tri[batch,:,2]
        # u_pred_convL,v_pred_convL = pred_convL[batch,:,1],pred_convL[batch,:,2]
        # u_pred_FNO,v_pred_FNO = pred_FNO[batch,:,1],pred_FNO[batch,:,2]
        # # u_truth,v_truth = data_truth[:,batch,1],data_truth[:,batch,2]
        # u_pred,v_pred = pred[:,batch,1],pred[:,batch,2]
        # u_pred_tri,v_pred_tri = pred_tri[:,batch,2],pred_tri[:,batch,2]
        # u_pred_convL,v_pred_convL = pred_convL[:,batch,2],pred_convL[:,batch,2]
        realsize_truth, EK_avsphr_truth,result_dict_truth = energy_specturm(u_truth,v_truth)
        realsize_pred, EK_avsphr_pred,result_dict_pred = energy_specturm(u_pred,v_pred)
        realsize_pred_tri, EK_avsphr_pred_tri,result_dict_pred_tri = energy_specturm(u_pred_tri,v_pred_tri)
        realsize_pred_convL, EK_avsphr_pred_convL,result_dict_pred_convL = energy_specturm(u_pred_convL,v_pred_convL)
        realsize_pred_FNO, EK_avsphr_pred_FNO,result_dict_pred_FNO = energy_specturm(u_pred_FNO,v_pred_FNO)
        fig= plt.figure(figsize=(5,5))
        plt.title(f"Kinetic Energy Spectrum -- {data_name}")
        plt.xlabel(r"k (wavenumber)")
        plt.ylabel(r"TKE of the k$^{th}$ wavenumber")
        print(realsize_truth)
        plt.loglog(np.arange(0,realsize_truth),((EK_avsphr_truth[0:realsize_truth] )),'k',label = "truth")
        plt.loglog(np.arange(0,realsize_pred),((EK_avsphr_pred[0:realsize_pred] )),'r',label = "pred (Ours))",alpha=0.6)
        plt.loglog(np.arange(0,realsize_pred_tri),((EK_avsphr_pred_tri[0:realsize_pred_tri] )),'b',label = "pred (Tri)",alpha=0.6)
        plt.loglog(np.arange(0,realsize_pred_convL),((EK_avsphr_pred_convL[0:realsize_pred_convL] )),'g',label = "pred (ConvL)",alpha=0.6)
        plt.loglog(np.arange(0,realsize_pred_FNO),((EK_avsphr_pred_FNO[0:realsize_pred_FNO] )),'y',label = "pred (FNO)",alpha=0.6)
        plt.legend()
        fig.savefig(f"{data_name}_energy_specturm_{batch}.png",dpi=300,bbox_inches='tight')
    return print("energy specturm plot done")

def plot_vorticity_correlation(data_name):
    data_truth = np.load(f"hr_target_{data_name}.npy")
    pred_tri = np.load(f"pred_tri_{data_name}.npy")
    pred_convL = np.load(f"pred_conv_{data_name}.npy")
    pred_FNO = np.load(f"pred_FNO_{data_name}.npy")
    pred = np.load(f"pred_NODE_{data_name}.npy")
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

    fig,axs = plt.subplots(1,1,figsize=(10,5))
    axs.set_xticks(np.arange(0,pred.shape[1],1))
    axs.plot(np.arange(0,pred.shape[1],1),correlations,color='k',label="Ours")
    axs.plot(np.arange(0,pred.shape[1],1),correlation_tri,color = 'b',label="Tri")
    axs.plot(np.arange(0,pred.shape[1],1),correlation_convL,color = 'g',label="ConvL")
    axs.plot(np.arange(0,pred.shape[1],1),correlation_FNO,color = 'y',label="FNO")
    # axs.scatter(np.arange(0,pred.shape[1],1),correlations,color='k')
    axs.scatter(np.arange(0,pred.shape[1],1),correlation_tri,color = 'b')
    axs.scatter(np.arange(0,pred.shape[1],1),correlation_convL,color = 'g')
    axs.scatter(np.arange(0,pred.shape[1],1),correlation_FNO,color = 'y')
    axs.legend()
    axs.set_ylabel("vorticity correlation")
    axs.set_xlabel("time")
    axs.set_title(f"vorticity correlation -- {data_name}")
    fig.savefig(f"vorticity_correlation_{data_name}.png",dpi=300,bbox_inches='tight')

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
    hr_target = np.load("hr_target_climate.npy")
    pred_tri = np.load("pred_tri_climate.npy")
    pred_conv = np.load("pred_conv_climate.npy")
    pred_FNO = np.load("pred_FNO_climate.npy")
    pred_NODE = np.load("pred_NODE_climate.npy")
    fig,ax = plt.subplots(5,3,figsize=(20,8))

    import seaborn
    for batch in [0,1,2,3,4]:
        for i in range(3):
            vmin = hr_target[batch,...,5:-5,5:-5].min()
            vmax = hr_target[batch,...,5:-5,5:-5].max()
            ax[0,i].imshow(hr_target[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = seaborn.cm.balance)
            ax[0,i].set_axis_off()
            ax[1,i].imshow(pred_tri[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = seaborn.cm.balance)
            ax[1,i].set_axis_off()
            ax[2,i].imshow(pred_FNO[batch,i*10,0,:,:],vmin=vmin,vmax=vmax,cmap = seaborn.cm.balance)
            ax[2,i].set_axis_off()
            ax[3,i].imshow(pred_conv[batch,i*10,0,5:-5,5:-5].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.balance)
            ax[3,i].set_axis_off()
            ax[4,i].imshow(pred_NODE[batch,i*10,0,5:-5,5:-5].T,vmin=vmin,vmax=vmax,cmap = seaborn.cm.balance)
            ax[4,i].set_axis_off()
        fig.savefig(f"Climate_baseline_{batch}.png",bbox_inches='tight')
import numpy as np 
import matplotlib.pyplot as plt
# plot_energy_specturm_overlay("DT")
# plot_RBC_comparision()
plot_energy_specturm_overlay("RBC")
# plot_vorticity_correlation("DT")
# plot_vorticity_correlation("RBC")