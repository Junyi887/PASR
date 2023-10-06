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
from src.data_loader import getData
import logging
import argparse

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def visualize(model,test1_loader,location =100,savedname = 'test.png'):
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            if batch_idx == 13:
                data,target = data.float().to(device) , target.float().to(device)
                output_quater1 = model(data,task_dt = 1.2,
                                    n_snapshot = 80,ode_step = 5,
                                    time_evol = True)[3,:,0,...].cpu().numpy()

                print(output_quater1.shape)
                target = target[3,1:,0,:,:].cpu().numpy()
                print(target.shape)
                data = data[3,0,:,:].cpu().numpy()
                plt.figure()
                plt.imshow(data)
                plt.axis('off')
                plt.savefig('figures/input' +".png",bbox_inches='tight',transparent = False)
                for i in range(output_quater1.shape[0]):
                    if i % 10 ==0:
                        plt.figure()
                        plt.imshow(target[i])
                        plt.axis('off')
                        plt.savefig('figures/target_' + str(i) +".png",bbox_inches='tight',transparent = False)
                        plt.figure()
                        plt.imshow(output_quater1[i])
                        plt.axis('off')
                        plt.savefig('figures/prediction_' + str(i) +".png",bbox_inches='tight',transparent = False)
                break
    return logging.info("visualization complete.")

def test_RFNE(model,test1_loader):
    eval_length = 4
    list_half1 = []
    list_half2 = []
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            data,target = data.to(device).float() , target.to(device).float()
            output_t = model(data,task_dt = args.task_dt/4,
                                n_snapshot = args.n_snapshot,ode_step = args.ode_step//4,
                                time_evol = True)
            output_x = model(data,task_dt = args.task_dt/4,
                                n_snapshot = 1,ode_step = args.ode_step,
                                time_evol = False)
            for i in range (target.shape[0]):
                for j in range (output_t.shape[1]):
                    RFNE_half1 = torch.norm(output_t[i,j,...]-target[i,j+1,...])/torch.norm(target[i,j+1,...])
                    list_half1.append(RFNE_half1)
                RFNE_half2 = torch.norm(output_x[i,0,...]-target[i,0,...])/torch.norm(target[i,0,...])
                list_half2.append(RFNE_half2)
    avg_half1 = torch.mean(torch.stack(list_half1),dim=0).item()
    avg_half2 = torch.mean(torch.stack(list_half2),dim=0).item()
    logging.info("Test RFNE complete.")
    return avg_half1,avg_half2

def test_RFNE_half(model,test1_loader):
    list_half1 = []
    list_half2 = []
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            data,target = data.to(device).float() , target.to(device).float()
            output_half1 = model(data,task_dt = args.task_dt,
                                n_snapshot = 1,ode_step = args.ode_step//2,
                                time_evol = True)
            output_half2 = model(data,task_dt = args.task_dt//2,
                                n_snapshot = 1,ode_step = args.ode_step,
                                time_evol = True)
            for i in range (target.shape[0]):
                RFNE_half1 = torch.norm(output_half1[i,0,...]-target[i,2,...])/torch.norm(target[i,2,...])
                RFNE_half2 = torch.norm(output_half2[i,0,...]-target[i,2,...])/torch.norm(target[i,2,...])
                list_half1.append(RFNE_half1)
                list_half2.append(RFNE_half2)
    avg_half1 = torch.mean(torch.stack(list_half1),dim=0).item()
    avg_half2 = torch.mean(torch.stack(list_half2),dim=0).item()
    logging.info("test RFNE complete.")
    return avg_half1,avg_half2

def test_RFNE_quater(model,test1_loader):
    list_quater1 = []
    # list_quater2 = []
    # list_quater3 = []
    # list_quater4 = []
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            data,target = data.to(device).float() , target.to(device).float()
            output_quater1 = model(data,task_dt = args.task_dt/4,
                                n_snapshot = 1,ode_step = 2,
                                time_evol = True)
            output_quater2 = model(data,task_dt = args.task_dt/4,
                                n_snapshot = 1,ode_step = 2,
                                time_evol = True)
            output_quater3 = model(data,task_dt = args.task_dt/4,
                                n_snapshot = 1,ode_step = 3,
                                time_evol = True)
            output_quater4 = model(data,task_dt = args.task_dt/4,
                                n_snapshot = 1,ode_step = 4,
                                time_evol = True)
            for i in range (target.shape[0]):
                RFNE_quater1 = torch.norm(output_quater1[i,0,...]-target[i,1,...])/torch.norm(target[i,1,...])
                RFNE_quater2 = torch.norm(output_quater2[i,0,...]-target[i,2,...])/torch.norm(target[i,2,...])
                RFNE_quater3 = torch.norm(output_quater3[i,0,...]-target[i,3,...])/torch.norm(target[i,3,...])
                RFNE_quater4 = torch.norm(output_quater4[i,0,...]-target[i,4,...])/torch.norm(target[i,4,...])
                
                list_quater1.append(RFNE_quater1)
                list_quater2.append(RFNE_quater2)
                list_quater3.append(RFNE_quater3)
                list_quater4.append(RFNE_quater4)

    avg_quater1 = torch.mean(torch.stack(list_quater1),dim=0).item()
    avg_quater2 = torch.mean(torch.stack(list_quater2),dim=0).item()
    avg_quater3 = torch.mean(torch.stack(list_quater3),dim=0).item()
    avg_quater4 = torch.mean(torch.stack(list_quater4),dim=0).item()

    return avg_quater1,avg_quater2,avg_quater3,avg_quater4

parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--model', type =str ,default= 'PASR_MLP_small')
parser.add_argument('--data', type =str ,default= 'rbc_diff_10IC')
parser.add_argument('--loss_type', type =str ,default= 'L2')
parser.add_argument('--scale_factor', type = int, default= 8)
parser.add_argument('--timescale_factor', type = int, default= 1)
parser.add_argument('--task_dt',type =float, default= 0.1)
parser.add_argument('--ode_step',type =int, default= 8)
parser.add_argument('--ode_method',type =str, default= "RK4")

parser.add_argument('--batch_size', type = int, default= 8)
parser.add_argument('--crop_size', type = int, default= 256, help= 'should be same as image dimension')
parser.add_argument('--epochs', type = int, default= 1)
parser.add_argument('--dtype', type = str, default= "float32")
parser.add_argument('--seed',type =int, default= 3407)


parser.add_argument('--n_snapshot',type =int, default= 80)
parser.add_argument('--down_method', type = str, default= "bicubic") # bicubic 
parser.add_argument('--upsampler', type = str, default= "pixelshuffle") # nearest+conv
parser.add_argument('--noise_ratio', type = float, default= 0.0)
parser.add_argument('--lr', type = float, default= 1e-4)
parser.add_argument('--lamb', type = float, default= 0.3)
parser.add_argument('--data_path',type = str,default = "../rbc_diff_IC/rbc_10IC")
args = parser.parse_args()
logging.info(args)
if __name__ == "__main__":
    if args.dtype =="float32":
        data_type = torch.float32
    else: 
        data_type = torch.float64

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_default_dtype(data_type)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _,val1_loader,_,test1_loader,test2_loader = getData(upscale_factor = args.scale_factor, timescale_factor= args.timescale_factor,batch_size = args.batch_size, crop_size = args.crop_size,data_path = args.data_path,num_snapshots = args.n_snapshot,data_name = args.data)
    mean = [0] 
    std = [1]
    model_list = {"PASR": PASR(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP":PASR_MLP(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP_G":PASR_MLP_G(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP_small":PASR_MLP(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP_G_small":PASR_MLP_G(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type)

    }
    model = torch.nn.DataParallel(model_list[args.model]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    savedpath = str(str(args.model) +
                "_data_" + str(args.data) + 
                "_crop_size_" + str(args.crop_size) +
                "_ode_step_" + str(args.ode_step) +
                "_ode_method_" + str(args.ode_method) +
                "_task_dt_" +  str(args.task_dt) + 
                "_num_snapshots_" + str(args.n_snapshot) +
                "_upscale_factor_" + str(args.scale_factor) +
                "_timescale_factor_" + str(args.timescale_factor) +
                "_loss_type_" + str(args.loss_type) +
                "_lamb_" + str(args.lamb)
                ) 
    checkpoint = torch.load("results/"+ "PASR_MLP_small_data_rbc_diff_10IC_crop_size_256_ode_step_5_ode_method_Euler_task_dt_1.2_num_snapshots_20_upscale_factor_8_timescale_factor_1_loss_type_L2_lamb_2.0.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    visualize(model,test1_loader)

    # print("RFNE_quater_interpolate --- test1 error: %.5f %%, test2 error: %.5f %%" % (err1_interpolate_q*100.0, err2_interpolate_q*100.0))
    # print("RFNE_quater_interpolate --- test3 error: %.5f %%, test4 error: %.5f %%" % (err3_interpolate_q*100.0, err4_interpolate_q*100.0))
    # print("RFNE_quater_extrapolate --- test1 error: %.5f %%, test2 error: %.5f %%" % (err1_extrapolate_q*100.0, err2_extrapolate_q*100.0))
    # print("RFNE_quater_extrapolate --- test3 error: %.5f %%, test4 error: %.5f %%" % (err3_extrapolate_q*100.0, err4_extrapolate_q*100.0))
