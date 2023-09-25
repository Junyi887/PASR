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

import torch
import numpy as np
from torch.utils import data
import matplotlib.pyplot as plt
import radialProfile

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
    args = parser.parse_args()
    checkpoint = torch.load(args.model_path)
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

    lr_input,hr_target,lr_input_tensor,hr_target_tensor = get_test_data("decay_turb",timescale_factor=4,num_snapshot = 20,in_channel=3,upscale_factor=4)

    lr_input2,hr_target2,lr_input_tensor2,hr_target_tensor2 = get_test_data("decay_turb",timescale_factor=2,num_snapshot = 40,in_channel=3,upscale_factor=4)

    pred = get_prediction(model,lr_input_tensor,hr_target_tensor,scale_factor = 4,in_channels = args.in_channels,task_dt = args.task_dt,n_snapshots = 20,ode_step=args.ode_step)
    pred2 = get_prediction(model,lr_input_tensor2,hr_target_tensor2,scale_factor = 4,in_channels = args.in_channels,task_dt = args.task_dt//2,n_snapshots = 40,ode_step=args.ode_step//2)
    RFNE,RFNE_mean,RFNE_avg = get_metric_RFNE(pred,hr_target_tensor)
    RFNE2,RFNE2_mean2,RFNE_avg2 = get_metric_RFNE(pred2,hr_target_tensor2)
    print(RFNE[-5,:,0])
    print(RFNE2[-5,:,0])
    print(RFNE[-11:-1,:,:].mean())
    print(RFNE2[-11:-1,:,:].mean())