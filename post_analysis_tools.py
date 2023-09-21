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
import torchaudio
from scipy.stats import pearsonr
from scipy.ndimage import zoom
CMAP = cmocean.cm.balance
CMAP = seaborn.cm.icefire


import torch
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import radialProfile


# DATA_INFO = {"decay_turb":['../Decay_Turbulence_small/train/Decay_turb_small_128x128_7202.h5', 0.02],
#             "burger2d": ["../Burgers_2D_small/train/Burgers2D_128x128_702.h5",0.001],
#             "rbc": ["../RBC_small/train/RBC_small_165_s2.h5",0.01]}

DATA_INFO = {"decay_turb":['../Decay_Turbulence_small/test/Decay_turb_small_128x128_79.h5', 0.02],
                 "burger2d": ["../Burgers_2D_small/test/Burgers2D_128x128_79.h5",0.001],
                 "rbc": ["../RBC_small/test/RBC_small_33_s2.h5",0.01]}

# MODEL_INFO = {"decay_turb":'results/PASR_MLP_small_data_Decay_turb_small_crop_size_256_ode_step_8_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L1_lamb_1.0_lr_0.0002_gamma_0.95_normalizaiton_Falsetensor([[1286]]).pt',
#             "burger2d": "results/PASR_MLP_small_data_Burger2D_small_crop_size_256_ode_step_8_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L1_lamb_0.3_lr_0.0005_gamma_0.95_normalizaiton_Falsetensor([[3624]]).pt",
#             "rbc": "results/PASR_MLP_small_data_rbc_small_crop_size_256_ode_step_8_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L1_lamb_0.3_lr_0.0003_gamma_0.95_normalizaiton_Falsetensor([[736]]).pt",
#             "rbc_p":"results/PASR_MLP_small_data_rbc_small_crop_size_256_ode_step_10_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L1_lamb_0.3_lr_0.0003_gamma_0.95_normalizaiton_False_8268.pt",
#             "decay_turb_p":"results/PASR_MLP_small_data_Decay_turb_small_crop_size_256_ode_step_10_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L1_lamb_0.3_lr_0.0002_gamma_0.95_normalizaiton_False_190.pt",
# }

MODEL_INFO = {"decay_turb":'results/PASR_MLP_small_data_Decay_turb_small_crop_size_256_ode_step_10_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L2_lamb_1.0_lr_0.0002_gamma_0.95_normalizaiton_False2865.pt',
            "decay_turb_physics":'results/PASR_MLP_small_data_Decay_turb_small_crop_size_256_ode_step_10_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L2_lamb_1.0_lr_0.0002_gamma_0.95_normalizaiton_False2865.pt',
            "rbc_physics": "results/PASR_MLP_small_data_rbc_small_crop_size_256_ode_step_10_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L2_lamb_1.0_lr_0.0002_gamma_0.95_normalizaiton_False7385.pt",
            "rbc": "results/PASR_MLP_small_data_rbc_small_crop_size_256_ode_step_10_ode_method_Euler_task_dt_4_num_snapshots_20_upscale_factor_4_timescale_factor_5_loss_type_L2_lamb_1.0_lr_0.0002_gamma_0.95_normalizaiton_False5486.pt"
}

def plot_for_comparision(data_name,pred,truth,lr_input,time_span,vmin,vmax,channel=0):
    lr_target = lr_input[time_span+1,channel]
    lr_input = lr_input[time_span,channel]
    error = np.abs(pred - truth)
    error = error[time_span,:,channel]
    pred = pred[time_span,:,channel]
    truth = truth[time_span,:,channel]
    lr_trajectory = np.zeros((pred.shape[0],lr_input.shape[0],lr_input.shape[1]))
    lr_trajectory[0] = lr_input
    lr_trajectory[-1] = lr_target

    figheight = 8 if data_name != "rbc" else 16
    figwidth = 2 if data_name != "rbc" else 1
    fig,axs = plt.subplots(4,truth.shape[0],figsize=(figwidth*truth.shape[0],figheight))
    for i in range(truth.shape[0]):
        img = axs[0,i].set_title(f"input t={(i*DATA_INFO[data_name][1]*time_span*5):.2f}") # 5 is the timescale factor
        axs[0,i].imshow(lr_trajectory[i],cmap=CMAP,vmin=vmin,vmax=vmax)
        axs[1,i].imshow(pred[i],cmap=CMAP,vmin=vmin,vmax=vmax)
        axs[1,i].set_title(f"pred t={(i*DATA_INFO[data_name][1]*time_span*5):.2f}") # 5 is the timescale factor
        axs[2,i].imshow(truth[i],cmap=CMAP,vmin=vmin,vmax=vmax)
        axs[2,i].set_title(f"truth t={(i*DATA_INFO[data_name][1]*time_span*5):.2f}")
        err = axs[3,i].imshow(error[i],cmap=CMAP,vmin=0,vmax=vmax)
        axs[3,i].set_title(f"error t={(i*DATA_INFO[data_name][1]*time_span*5):2f}")
    # plt.colorbar(err,ax=axs[3,-1],shrink = 0.5,extend = 'both')
    # plt.colorbar(img,ax = [axs[0,-1],axs[1,-1],axs[2,-1]],shrink = 0.5,extend = 'both')
    for ax in axs:
        for a in ax:
            a.axis('off') 
    fig.savefig(f"figures/absolute_error_{data_name}.png",dpi=300,bbox_inches='tight')

    fig,axs = plt.subplots(4,truth.shape[0]//5,figsize=(figwidth*truth.shape[0]//5,figheight))
    for i in range(truth.shape[0]//5):
        axs[0,i].set_title(f"input t={(i*5*DATA_INFO[data_name][1]*time_span*5):.2f}")
        axs[0,i].imshow(lr_trajectory[i*5],cmap=CMAP,vmin=vmin,vmax=vmax)
        axs[1,i].imshow(pred[i*5],cmap=CMAP,vmin=vmin,vmax=vmax)
        axs[1,i].set_title(f"pred t={(i*DATA_INFO[data_name][1]*time_span*5*5):.2f}")
        axs[2,i].imshow(truth[i*5],cmap=CMAP,vmin=vmin,vmax=vmax)
        axs[2,i].set_title(f"truth t={(i*DATA_INFO[data_name][1]*time_span*5*5):.2f}")
        axs[3,i].imshow(error[i*5],cmap=CMAP,vmin=0,vmax=vmax)
        axs[3,i].set_title(f"error t={(i*DATA_INFO[data_name][1]*time_span*5*5):.2f}")
    # plt.colorbar(err,ax=axs[3,-1],shrink = 0.5,extend = 'both')
    # plt.colorbar(img,ax = [axs[0,-1],axs[1,-1],axs[2,-1]],shrink = 0.5,extend = 'both')
    for ax in axs:
        for a in ax:
            a.axis('off') 
    fig.savefig(f"figures/absolute_error_{data_name}_every5.png",dpi=300,bbox_inches='tight')
    return print("absolute error plot done")

def plot_vorticity_correlation(data_name,pred,reference_data):
    correlations = np.zeros(pred.shape[0])
    for t in range(pred.shape[0]):
        ref_flat = reference_data[t].flatten()
        gen_flat = pred[t].flatten()
        corr, _ = pearsonr(ref_flat, gen_flat)
        correlations[t] = corr
    fig,axs = plt.subplots(1,1,figsize=(20,10))
    axs.set_xticks(np.arange(0,w_truth.shape[0],20))
    axs.plot(np.arange(0,w_truth.shape[0]),correlations,'-.')
    axs.set_ylabel("vorticity correlation")
    axs.set_xlabel("time")
    axs.set_title(f"vorticity correlation -- {data_name}")
    fig.savefig(f"figures/vorticity_correlation_{data_name}.png",dpi=300,bbox_inches='tight')

    return print("voritcity correlation plot done")

def energy_specturm(u,v):
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from math import sqrt
    c  = sqrt(1.4);
    Ma = 0.1;
    U0 = 1.0; 
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

def plot_PDF(data_name,pred,truth,lr_input):
    import numpy as np
    from scipy.interpolate import UnivariateSpline
    from matplotlib import pyplot as plt
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # # Flatten the prediction and truth arrays
    # s_pred = pred.flatten()
    # s_truth = truth.flatten()
    # # Plot the KDE for rediction and truth
    # seaborn.kdeplot(s_pred, ax=ax,label="Prediction",log_scale=True)
    # seaborn.kdeplot(s_truth, label="Truth",ax=ax,log_scale=True)
    # # Labeling and legend
    # ax.set_xlabel("Normalized Vorticity")
    # ax.legend()
    # # Save the figure
    # fig.savefig("figures/PDF_" + data_name + ".png")
    # s_pred3 = lr_input[:,:,0].flatten()
    # n3 = s_pred3.shape[0]
    s_pred = pred.flatten()
    s_truth = truth.flatten()
    n = 3000
    std = np.std(s_pred)
    std2 = np.std(s_truth)
    p, x = np.histogram(s_pred/std, bins=n) # bin it into n = N//10 bins
    p2,x2 = np.histogram(s_truth/std2, bins=n)
    # p3,x3 = np.histogram(s_pred3, bins=n,density = True)
#    x3 = x3[:-1] + (x3[1] - x3[0])/2   # convert bin edges to centers
    x2 = x2[:-1] + (x2[1] - x2[0])/2   # convert bin edges to centers
    x = x[:-1] + (x[1] - x[0])/2   # convert bin edges to centers

    f = UnivariateSpline(x, p, s=n)
    f2 = UnivariateSpline(x2, p2, s=n)
    fig,ax = plt.subplots(1,1,figsize = (10,10))
    ax.plot(x,f(x),'k',label = "Prediction")
    ax.plot(x2,f2(x2),'r',label = "Ground Truth")
    # plt.plot(x3,f3(x3))
    ax.set_yscale('log')
    ax.set_xlim(-30,30)
    ax.set_xlabel('Normalized vorticity')
    ax.set_ylabel('PDF')
    ax.legend(['Prediction','Ground Truth'])
    ax.set_title(data_name)
    fig.savefig("figures/PDF_"+data_name+".png")
    return print("PDF plot done")

def get_data_scale(data_name):
    f = h5py.File(DATA_INFO[data_name][0],'r')
    w = f['tasks']['vorticity'][()] if data_name != "burger2d" else None
    u = f['tasks']['u'][()]
    v = f['tasks']['v'][()]
    print(f"************{data_name}************")
    print(f"max w: {np.max(w):.2f}, min w: {np.min(w):.2f}") if data_name != "burger2d" else None
    print(f"mean w: {np.mean(w):4f}, std w: {np.std(w):.4f}") if data_name != "burger2d" else None
    print(f"max u: {np.max(u):.4f}, min u: {np.min(u):.4f}")
    print(f"mean u: {np.mean(u):.4f}, std u: {np.std(u):.4f}")
    print(f"max v: {np.max(v):.4f}, min v: {np.min(v):.4f}")
    print(f"mean v: {np.mean(v):.4f}, std v: {np.std(v):.4f}")
    # dic = {"max w":np.max(w),"min w":np.min(w),"mean w":np.mean(w),"std w":np.std(w),"max u":np.max(u),"min u":np.min(u),"mean u":np.mean(u),"std u":np.std(u),"max v":np.max(v),"min v":np.min(v),"mean v":np.mean(v),"std v":np.std(v)}
    f.close()
    return None



def plot_data(data_name = "burger2d",row=10,col=10,timescale_factor =10 ,in_channel=2,vmin=-1,vmax=1):
    f = h5py.File(DATA_INFO[data_name][0],'r')
    if data_name == "rbc":
        print(f['scales/sim_time'][()])
        t = int(f['scales/sim_time'][0])
    else:
        print(f['tasks']['t'][()])
        t = np.ceil(f['tasks']['t'][0])
    print(f['tasks']['u'].shape)
    if data_name == "burger2d" and in_channel ==1:
        raise ValueError("Burger2D doesnt have vorticity channel")
    else:
        if in_channel == 1:
            w = f['tasks']['vorticity'][()]
            fig,axs = plt.subplots(row,col,figsize=(20,20))
            i = 0
            for ax in axs:
                for a in ax:
                    a.axis('off')
                    a.imshow(w[i*timescale_factor],cmap=CMAP,vmin=vmin,vmax=vmax)
                    a.set_title(f"t={t + i*timescale_factor*DATA_INFO[data_name][1]:.4}")
                    i+=1
        elif in_channel == 2:
            u = f['tasks']['u'][()]
            v = f['tasks']['v'][()]
            fig,axs = plt.subplots(row,col,figsize=(20,20))
            i = 0
            for ax in axs:
                for a in ax:
                    a.axis('off')
                    a.imshow(u[i*timescale_factor],cmap=CMAP,vmin=vmin[0],vmax=vmax[0])
                    a.set_title(f"t={t + i*timescale_factor*DATA_INFO[data_name][1]:.4}")
                    i+=1
            fig,axs = plt.subplots(row,col,figsize=(20,20))
            i = 0
            for ax in axs:
                for a in ax:
                    a.axis('off')
                    a.imshow(v[i*timescale_factor],cmap=CMAP,vmin=vmin[1],vmax=vmax[1])
                    a.set_title(f"t={t + i*timescale_factor*DATA_INFO[data_name][1]:.4}")
                    i+=1
        elif in_channel == 3:
            u = f['tasks']['u'][()]
            v = f['tasks']['v'][()]
            w = f['tasks']['vorticity'][()]
            fig,axs = plt.subplots(row,col,figsize=(20,20))
            i = 0
            for ax in axs:
                for a in ax:
                    a.axis('off')
                    a.imshow(u[i*timescale_factor],cmap=CMAP,vmin=vmin[0],vmax=vmax[0])
                    a.set_title(f"t={t + i*timescale_factor*DATA_INFO[data_name][1]:.4}")
                    i+=1
            fig,axs = plt.subplots(row,col,figsize=(20,20))
            i = 0
            for ax in axs:
                for a in ax:
                    a.axis('off')
                    a.imshow(v[i*timescale_factor],cmap=CMAP,vmin=vmin[1],vmax=vmax[1])
                    a.set_title(f"t={t + i*timescale_factor*DATA_INFO[data_name][1]:.4}")
                    i+=1
            fig,axs = plt.subplots(row,col,figsize=(20,20))
            i = 0
            for ax in axs:
                for a in ax:
                    a.axis('off')
                    a.imshow(w[i*timescale_factor],cmap=CMAP,vmin=vmin[2],vmax=vmax[2])
                    a.set_title(f"t={t + i*timescale_factor*DATA_INFO[data_name][1]:.4}")
                    i+=1
    f.close()
    fig.savefig(f"figures/{data_name}_data.png",dpi=300,bbox_inches='tight')
    return print("visualization done")


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

def get_test_data(data_name,timescale_factor = 10,num_snapshot = 10,in_channel=1,upscale_factor=4):

    with h5py.File(DATA_INFO[data_name][0],'r') as f:
        w_truth = f['tasks']['vorticity'][()] if in_channel ==1 or in_channel ==3 else None
        u_truth = f['tasks']['u'][()]
        v_truth = f['tasks']['v'][()]
    final_index = (u_truth.shape[0]-1)//timescale_factor
    idx_matrix = generate_test_matrix(num_snapshot +1 , final_index)*timescale_factor    
    print(idx_matrix[1:4])
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

def psnr(true, pred):
    mse = torch.mean((true - pred) ** 2)
    if mse == 0:
        return float(9999)
    max_value = torch.max(true)
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    if psnr.isnan() or psnr.isinf():
        return float(0)
    return psnr

def ssim(true, pred):
    from torchmetrics import StructuralSimilarityIndexMeasure
    ssim = StructuralSimilarityIndexMeasure()
    return ssim(true, pred)

def get_prediction(model_dic,lr_input_tensor,hr_target_tensor,scale_factor,in_channels,task_dt,n_snapshot,ode_step):
    model = PASR_MLP(upscale=scale_factor, in_chans=in_channels, img_size=256, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler="pixelshuffle", resi_conv='1conv',mean=[0],std=[1])
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(model_dic)
    model_dic = checkpoint['model_state_dict']
    model.load_state_dict(model_dic)
    model.eval()
    with torch.no_grad():
        pred0 = model(lr_input_tensor.float().cuda(),task_dt = task_dt,n_snapshot = n_snapshot,ode_step = ode_step,time_evol = False)
        pred = model(lr_input_tensor.float().cuda(),task_dt = task_dt,n_snapshot = n_snapshot,ode_step = ode_step,time_evol = True)
        pred = torch.cat((pred0,pred),dim=1)
    return pred.detach().cpu().numpy()


def get_metric_RFNE(pred,truth):
    RFNE = torch.norm(pred - hr_target_tensor.float().cuda(), p=2, dim=(1, 2, 3)) / torch.norm(hr_target_tensor.float().cuda(), p=2, dim=(1, 2, 3))
    avg_RFNE = RFNE.mean().item()
    cum_RFNE = torch.norm(pred.flatten()-truth.flatten().float().cuda(),p=2)/torch.norm(truth.flatten().float().cuda(),p=2)
    RFNE = torch.norm(pred - truth.float().cuda(), p=2, dim=(1, 2, 3)) / torch.norm(truth.float().cuda(), p=2, dim=(1, 2, 3))
    print(f"averaged RFNE {avg_RFNE}")
    print(f"cumulative RFNE {cum_RFNE.item()}")
    return avg_RFNE,cum_RFNE.item()

def plot_spectrogram(u_truth,v_truth,u_pred,v_pred):
    fig,ax = plt.subplots(1,2,figsize=(5,8))
    energy_truth = 0.5*(u_truth**2 + v_truth**2) # (t,h,w)
    E_t = energy_truth.reshape(energy_truth.shape[0],-1)
    E_t = E_t.mean(axis=1)
    spectrogram = torchaudio.transforms.Spectrogram(win_length=5, hop_length=10, power=1).cuda()
    wfs = torch.Tensor(E_t.T)
    wfs = wfs.cuda().float()
    wfs = spectrogram(wfs)
    ax[0].imshow(wfs.cpu().numpy(),cmap=CMAP,vmin =0)
    ax[0].set_title("truth")
    energy_pred = 0.5*(u_pred**2 + v_pred**2) # (t,h,w)
    E_t = energy_pred.reshape(energy_pred.shape[0],-1)
    E_t = E_t.mean(axis=1)
    spectrogram = torchaudio.transforms.Spectrogram(win_length=5, hop_length=10, power=1).cuda()
    wfs = torch.Tensor(E_t.T)
    wfs = wfs.cuda().float()
    wfs = spectrogram(wfs)
    ax[1].imshow(wfs.cpu().numpy(),cmap=CMAP,vmin =0)
    ax[1].set_title("pred")
    ax[1].set_ylabel("time")
    return print("spectrogram plot done")

def plot_energy_specturm(u_truth,v_truth,u_pred,v_pred,data_name):
    realsize_truth, EK_avsphr_truth,result_dict_truth = energy_specturm(u_truth,v_truth)
    realsize_pred, EK_avsphr_pred,result_dict_pred = energy_specturm(u_pred,v_pred)
    fig= plt.figure(figsize=(5,5))
    plt.title(f"Kinetic Energy Spectrum -- {data_name} dt = {DATA_INFO[data_name][1]*20*5}")
    plt.xlabel(r"k (wavenumber)")
    plt.ylabel(r"TKE of the k$^{th}$ wavenumber")
    print(realsize_truth)
    plt.loglog(np.arange(0,realsize_truth),((EK_avsphr_truth[0:realsize_truth] )),'k',label = "truth")
    plt.loglog(np.arange(0,realsize_pred),((EK_avsphr_pred[0:realsize_pred] )),'r',label = "pred")
    plt.ylim(1e-7,1)
    plt.legend()
    fig.savefig(f"figures/{data_name}_energy_specturm.png",dpi=300,bbox_inches='tight')

def plot_energy_specturm_phyLoss(u_truth,v_truth,u_pred,v_pred,u_pred_p,v_pred_p,data_name):
    realsize_truth, EK_avsphr_truth,result_dict_truth = energy_specturm(u_truth,v_truth)
    realsize_pred, EK_avsphr_pred,result_dict_pred = energy_specturm(u_pred,v_pred)
    realsize_pred_p, EK_avsphr_pred_p,result_dict_pred_p = energy_specturm(u_pred_p,v_pred_p)
    fig= plt.figure(figsize=(5,5))
    plt.title(f"Kinetic Energy Spectrum -- {data_name} dt = {DATA_INFO[data_name][1]*20*5}")
    plt.xlabel(r"k (wavenumber)")
    plt.ylabel(r"TKE of the k$^{th}$ wavenumber")
    print(realsize_truth)
    plt.loglog(np.arange(0,realsize_truth),((EK_avsphr_truth[0:realsize_truth] )),'k',label = "truth")
    plt.loglog(np.arange(0,realsize_pred),((EK_avsphr_pred[0:realsize_pred] )),'r',label = "pred")
    plt.loglog(np.arange(0,realsize_pred_p),((EK_avsphr_pred_p[0:realsize_pred_p] )),'b',label = "pred (physics constraint)")
    plt.legend()
    fig.savefig(f"figures/{data_name}_energy_specturm.png",dpi=300,bbox_inches='tight')
    return print("energy specturm plot done")

if __name__ == "__main__":
    get_data_scale("rbc")
    plot_data(data_name = "rbc",timescale_factor =5 ,in_channel=1,vmin=-10,vmax=10,row=3,col=10)
    get_data_scale("burger2d")
    plot_data(data_name = "burger2d",timescale_factor =5 ,in_channel=2,vmin=[-1.2,-1.6],vmax=[2.8,1],row=10,col=10)
    get_data_scale("decay_turb")
    plot_data(data_name = "decay_turb",timescale_factor =5 ,in_channel=1,vmin=-5.7,vmax=7.82,row=10,col=10)
################## rbc ##################
lr_input,hr_target,lr_input_tensor,hr_target_tensor = get_test_data("rbc",timescale_factor = 5,num_snapshot = 20,in_channel=3,upscale_factor=4)
pred = get_prediction(MODEL_INFO["rbc"],lr_input_tensor,hr_target_tensor,scale_factor=4,in_channels=3,task_dt=4,n_snapshot=20,ode_step=10)
pred_p = get_prediction(MODEL_INFO["rbc_physics"],lr_input_tensor,hr_target_tensor,scale_factor=4,in_channels=3,task_dt=4,n_snapshot=20,ode_step=10)
u_truth = hr_target[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
v_truth = hr_target[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
u_pred = pred[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
v_pred = pred[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
u_pred_p = pred_p[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
v_pred_p = pred_p[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
plot_energy_specturm_phyLoss(u_truth[200:],v_truth[200:],u_pred[200:],v_pred[200:],u_pred_p[200:],v_pred_p[200:],"rbc")
plot_energy_specturm(u_truth[200:],v_truth[200:],u_pred[200:],v_pred[200:],"rbc")
rnfe1,rfne2 = get_metric_RFNE(torch.from_numpy(pred).float().cuda(),hr_target_tensor)
rfne1_p,rfne2_p = get_metric_RFNE(torch.from_numpy(pred_p).float().cuda(),hr_target_tensor)
print(f"averaged RFNE {rnfe1}, cumulative RFNE {rfne2} for data rbc")
print(f"averaged RFNE {rfne1_p}, cumulative RFNE {rfne2_p} for data rbc (physics constraint)")
w_truth = hr_target[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
w_pred = pred[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
w_pred_p = pred_p[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
plot_vorticity_correlation("rbc",w_pred,w_truth)
plot_vorticity_correlation("rbc_p",w_pred_p,w_truth)

# plot_for_comparision("rbc",pred,hr_target,lr_input,time_span=10,vmin=-10,vmax=10)
# plot_PDF("rbc",pred,hr_target,lr_input)

################## decay_turb ##################

lr_input,hr_target,lr_input_tensor,hr_target_tensor = get_test_data("decay_turb",timescale_factor = 5,num_snapshot = 20,in_channel=3,upscale_factor=4)
pred = get_prediction(MODEL_INFO["decay_turb"],lr_input_tensor,hr_target_tensor,scale_factor=4,in_channels=3,task_dt=4,n_snapshot=20,ode_step=10)
pred_p = get_prediction(MODEL_INFO["decay_turb_physics"],lr_input_tensor,hr_target_tensor,scale_factor=4,in_channels=3,task_dt=4,n_snapshot=20,ode_step=10)
u_truth = hr_target[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
v_truth = hr_target[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
u_pred = pred[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
u_pred_p = pred_p[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
v_pred = pred[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
v_pred_p = pred_p[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
plot_energy_specturm(u_truth[200:],v_truth[200:],u_pred[200:],v_pred[200:],'decay_turb')
plot_energy_specturm_phyLoss(u_truth[200:],v_truth[200:],u_pred[200:],v_pred[200:],u_pred_p[200:],v_pred_p[200:],'decay_turb')
rnfe1,rfne2 = get_metric_RFNE(torch.from_numpy(pred).float().cuda(),hr_target_tensor)
rfne1_p,rfne2_p = get_metric_RFNE(torch.from_numpy(pred_p).float().cuda(),hr_target_tensor)
print(f"averaged RFNE {rnfe1}, cumulative RFNE {rfne2} for data decay_turb")
print(f"averaged RFNE {rfne1_p}, cumulative RFNE {rfne2_p} for data decay_turb (physics constraint)")

w_truth = hr_target[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
w_pred_p = pred_p[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
w_pred = pred[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
plot_vorticity_correlation("decay_turb",w_pred,w_truth)
plot_vorticity_correlation("decay_turb_p",w_pred_p,w_truth)
# plot_for_comparision("decay_turb",pred,hr_target,lr_input,time_span=10,vmin=-5.7,vmax=7.82)
# plot_PDF("decay_turb",pred,hr_target,lr_input)


################## burger2d ##################
# model_dic = MODEL_INFO["burger2d"]
# lr_input,hr_target,lr_input_tensor,hr_target_tensor = get_test_data("burger2d",timescale_factor = 5,num_snapshot = 20,in_channel=2,upscale_factor=4)
# pred = get_prediction(model_dic,lr_input_tensor,hr_target_tensor,scale_factor=4,in_channels=2,task_dt=4,n_snapshot=20,ode_step=8)
# u_truth = hr_target[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# v_truth = hr_target[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# u_pred = pred[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# v_pred = pred[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# plot_energy_specturm(u_truth[200:],v_truth[200:],u_pred[200:],v_pred[200:],'burger2d')
# rnfe1,rfne2 = get_metric_RFNE(torch.from_numpy(pred).float().cuda(),hr_target_tensor)
# print(f"averaged RFNE {rnfe1}, cumulative RFNE {rfne2} for data burger2d")
# plot_for_comparision("burger2d",pred,hr_target,lr_input,time_span=5,vmin=-1.2,vmax=2.8)
# plot_PDF("burger2d",pred,hr_target,lr_input)
# print(lr_input.shape)

# model_dic = MODEL_INFO["rbc_p"]
# lr_input,hr_target,lr_input_tensor,hr_target_tensor = get_test_data("rbc",timescale_factor = 5,num_snapshot = 20,in_channel=3,upscale_factor=4)
# pred = get_prediction(model_dic,lr_input_tensor,hr_target_tensor,scale_factor=4,in_channels=3,task_dt=4,n_snapshot=20,ode_step=8)
# u_truth = hr_target[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# v_truth = hr_target[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# u_pred = pred[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# v_pred = pred[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# plot_energy_specturm(u_truth,v_truth,u_pred,v_pred,"rbc")
# rnfe1,rfne2 = get_metric_RFNE(torch.from_numpy(pred).float().cuda(),hr_target_tensor)
# print(f"averaged RFNE {rnfe1}, cumulative RFNE {rfne2} for data rbc")
# w_truth = hr_target[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# w_pred = pred[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
# plot_vorticity_correlation("rbc",w_pred,w_truth)
# plot_for_comparision("rbc",pred,hr_target,lr_input,time_span=10,vmin=-10,vmax=10)
# plot_PDF("rbc",pred,hr_target,lr_input)


model_dic = MODEL_INFO["decay_turb"]
lr_input,hr_target,lr_input_tensor,hr_target_tensor = get_test_data("decay_turb",timescale_factor = 1,num_snapshot = 100,in_channel=3,upscale_factor=4)
pred = get_prediction(model_dic,lr_input_tensor,hr_target_tensor,scale_factor=4,in_channels=3,task_dt=4,n_snapshot=100,ode_step=2)
u_truth = hr_target[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
v_truth = hr_target[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
u_pred = pred[:,:,1].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
v_pred = pred[:,:,2].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
plot_energy_specturm(u_truth,v_truth,u_pred,v_pred,'decay_turb')
rnfe1,rfne2 = get_metric_RFNE(torch.from_numpy(pred).float().cuda(),hr_target_tensor)
print(f"averaged RFNE {rnfe1}, cumulative RFNE {rfne2} for data decay_turb")
w_truth = hr_target[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
w_pred = pred[:,:,0].transpose(0,1,2,3).reshape(-1,pred.shape[-2],pred.shape[-1])
plot_vorticity_correlation("decay_turb",w_pred,w_truth)
plot_for_comparision("decay_turb",pred,hr_target,lr_input,time_span=10,vmin=-5.7,vmax=7.82)
plot_PDF("decay_turb",pred,hr_target,lr_input)