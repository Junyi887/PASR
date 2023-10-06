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


def plot_energy_specturm(u_truth,v_truth,u_pred,v_pred,data_name):
    realsize_truth, EK_avsphr_truth,result_dict_truth = energy_specturm(u_truth,v_truth)
    realsize_pred, EK_avsphr_pred,result_dict_pred = energy_specturm(u_pred,v_pred)
    fig= plt.figure(figsize=(5,5))
    plt.title(f"Kinetic Energy Spectrum -- {data_name}")
    plt.xlabel(r"k (wavenumber)")
    plt.ylabel(r"TKE of the k$^{th}$ wavenumber")
    print(realsize_truth)
    plt.loglog(np.arange(0,realsize_truth),((EK_avsphr_truth[0:realsize_truth] )),'k',label = "truth")
    plt.loglog(np.arange(0,realsize_pred),((EK_avsphr_pred[0:realsize_pred] )),'r',label = "pred")
    plt.ylim(1e-7,1)
    plt.legend()
    fig.savefig(f"{data_name}_energy_specturm.png",dpi=300,bbox_inches='tight')

DATA_INFO = {"decay_turb":['../Decay_Turbulence_small/test/Decay_turb_small_128x128_79.h5', 0.02],
                 "burger2d": ["../Burgers_2D_small/test/Burgers2D_128x128_79.h5",0.001],
                 "rbc": ["../RBC_small/test/RBC_small_33_s2.h5",0.01]}

def get_test_data(data_name,timescale_factor = 10,num_snapshot = 10,in_channel=1,upscale_factor=4):

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


                 
def get_prediction(model,lr_input_tensor,hr_target_tensor,scale_factor,in_channels,task_dt,n_snapshots,ode_step):

    model.eval()
    with torch.no_grad():
        pred0 = model(lr_input_tensor.float().cuda(),task_dt = task_dt,n_snapshots = n_snapshots,ode_step = ode_step,time_evol = False)
        pred = model(lr_input_tensor.float().cuda(),task_dt = task_dt,n_snapshots = n_snapshots,ode_step = ode_step,time_evol = True)
        pred = torch.cat((pred0,pred),dim=1)
    return pred.detach().cpu()


def get_metric_RFNE(pred,truth):
    # input should be tensor
    print("predshape",pred.shape)
    RFNE = torch.norm(pred.float().cuda()- truth.float().cuda(), p=2, dim=(-1, -2)) / torch.norm(truth.float().cuda(), p=2, dim=(-1, -2))
    avg_RFNE = RFNE.mean().item()
    cum_RFNE = torch.norm(pred.flatten().float().cuda()-truth.flatten().float().cuda(),p=2)/torch.norm(truth.flatten().float().cuda(),p=2)
    print(f"averaged RFNE {avg_RFNE}")
    print(f"cumulative RFNE {cum_RFNE.item()}")
    return RFNE.detach().cpu().numpy(),avg_RFNE,cum_RFNE.item()

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PASR')
    parser.add_argument("--model_path", type=str, default="results/pre_trained_PASR_MLP_small_data_Decay_turb_small_0.pt", help="path to model")
    parser.add_argument("--test_data_name", type=str, default="decay_turb", help="decay_turb, burger2d, rbc")
    parsed_args = parser.parse_args()
    checkpoint = torch.load(parsed_args.model_path)
    model_state = checkpoint['model_state_dict']
    config = checkpoint['config']
    args = argparse.Namespace()
    args.__dict__.update(config)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.normalization == "True":
        mean,std = getNorm(args)
        mean = [mean]
        std = [std]
    else:
        mean = [0]
        std = [1]
    if args.data =="Decay_turb_small": 
        image = [128,128]
    elif args.data =="rbc_small":
        image = [256,64]
    elif args.data =="Burger2D_small":
        image = [128,128]
    model_list = {
            "PASR_small":PASR(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,num_ode_layers = args.ode_layer,time_update = args.time_update,ode_kernel_size = args.ode_kernel,ode_padding = args.ode_padding),
             "PASR_MLP_small":PASR_MLP(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std),
            "PASR_MLP":PASR_MLP(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std),
            "PASR_MLP_G":PASR_MLP_G(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,gating_layers=args.gating_layers,gating_method=args.gating_method),
            "PASR_MLP_small":PASR_MLP(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std),
            "PASR_MLP_G_small":PASR_MLP_G(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,gating_layers=args.gating_layers,gating_method=args.gating_method),
    }
    model = model_list[args.model]
    model = torch.nn.DataParallel(model).to(device)
    model.load_state_dict(model_state)
    lr_input,hr_target,lr_input_tensor,hr_target_tensor = get_test_data(parsed_args.test_data_name,timescale_factor=4,num_snapshot = 20,in_channel=3,upscale_factor=4)
    pred = get_prediction(model,lr_input_tensor,hr_target_tensor,scale_factor = 4,in_channels = args.in_channels,task_dt = args.task_dt,n_snapshots = 20,ode_step=args.ode_step)
    # pred = pred.flatten(0,1)
    pred_numpy = pred.numpy()
    B,T,C,H,W = pred_numpy.shape
    B_lr,C_lr,H_lr,W_lr = lr_input.shape
    u_truth = hr_target[4,:,1,:,:]
    v_truth = hr_target[4,:,2,:,:]
    u_pred = pred_numpy[4,:,1,:,:]
    v_pred = pred_numpy[4,:,2,:,:]
    plot_energy_specturm(u_truth,v_truth,u_pred,v_pred,"DT")
    # np.save(f"results_{parsed_args.test_data_name}.npy",{
    #     "pred_numpy":pred_numpy,
    #     "hr_target":hr_target,
    #     "lr_input":lr_input
    # })
    # if parsed_args.test_data_name == 'rbc':
    #     xx,yy = 4,1
    #     vmin,vmax = -30,32
    # else:
    #     xx,yy = 2,2
    #     vmin,vmax = -7.9,7.2
    # fig,a = plt.subplots(1,1,figsize = (xx,yy))
    # counts = 0
    # for i in range(B-12):
    #     for j in range (T-1):
    #         a.axis("off")
    #         a.imshow(pred_numpy[i+3,j,0,:,:],cmap = seaborn.cm.icefire,vmin = vmin,vmax = vmax)
    #         fig.savefig(f"frames/{parsed_args.test_data_name}_prediction_{counts}.png",bbox_inches = "tight",dpi = 400)
    #         counts +=1 
    
    # fig,a = plt.subplots(1,1,figsize = (xx,yy))
    # counts = 0
    # for i in range(B-12):
    #     for j in range (T-1):
    #         a.axis("off")
    #         a.imshow(hr_target[i+3,j,0,:,:],cmap = seaborn.cm.icefire,vmin = vmin,vmax = vmax)
    #         fig.savefig(f"frames/{parsed_args.test_data_name}_target_{counts}.png",bbox_inches = "tight",dpi = 400)
    #         counts +=1 

    # fig,a = plt.subplots(1,1,figsize = (xx,yy))
    # counts = 0
    # for i in tqdm(range(B_LR-12)):
    #     a.axis("off")
    #     a.imshow(lr_input[i+3,0,:,:],cmap = seaborn.cm.icefire,vmin = vmin,vmax = vmax)
    #     fig.savefig(f"frames/{parsed_args.test_data_name}_LR_{counts}.png",bbox_inches = "tight",dpi = 400)
    #     counts +=1 

