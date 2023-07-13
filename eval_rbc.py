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
from src.utli import *
from src.data_loader import getData
import logging
import argparse

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def visualize(model,test1_loader,location =100,savedname = 'test.png'):
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            if batch_idx == 4:
                data,target = data.float().to(device) , target.float().to(device)
                output_quater1 = model(data,task_dt = args.task_dt/4,
                                    n_snapshot = 4*args.n_snapshot,ode_step = 2,
                                    time_evol = True)[0,:,0,location,location]
                target = target[0,1:,0,location,location]
                plt.figure()
                plt.plot(output_quater1.cpu().numpy(),marker = "o",label = "prediction")
                plt.plot(target.cpu().numpy(),marker = "o",label = "truth")
                plt.xlabel("time")
                plt.ylabel("pixel value")
                plt.legend()
                plt.savefig('figures/' + savedname)
                break
    return logging.info("visualization complete.")

def test_RFNE(model,test1_loader):
    eval_length = 4
    list_half1 = []
    list_half2 = []
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            data,target = data.to(device).float() , target.to(device).float()
            output_t = model(data,task_dt = args.task_dt//4,
                                n_snapshot = eval_length,ode_step = 2,
                                time_evol = True)
            output_x = model(data,task_dt = args.task_dt//4,
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

# def test_RFNE_half(model,test1_loader):
#     list_half1 = []
#     list_half2 = []
#     with torch.no_grad():
#         for batch_idx,(data,target) in enumerate(test1_loader):
#             data,target = data.to(device).float() , target.to(device).float()
#             output_half1 = model(data,task_dt = args.task_dt,
#                                 n_snapshot = 1,ode_step = args.ode_step//2,
#                                 time_evol = True)
#             output_half2 = model(data,task_dt = args.task_dt//2,
#                                 n_snapshot = 1,ode_step = args.ode_step,
#                                 time_evol = True)
#             for i in range (target.shape[0]):
#                 RFNE_half1 = torch.norm(output_half1[i,0,...]-target[i,2,...])/torch.norm(target[i,2,...])
#                 RFNE_half2 = torch.norm(output_half2[i,0,...]-target[i,2,...])/torch.norm(target[i,2,...])
#                 list_half1.append(RFNE_half1)
#                 list_half2.append(RFNE_half2)
#     avg_half1 = torch.mean(torch.stack(list_half1),dim=0).item()
#     avg_half2 = torch.mean(torch.stack(list_half2),dim=0).item()
#     logging.info("test RFNE complete.")
#     return avg_half1,avg_half2

# def test_RFNE_quater(model,test1_loader):
#     list_quater1 = []
#     # list_quater2 = []
#     # list_quater3 = []
#     # list_quater4 = []
#     with torch.no_grad():
#         for batch_idx,(data,target) in enumerate(test1_loader):
#             data,target = data.to(device).float() , target.to(device).float()
#             output_quater1 = model(data,task_dt = args.task_dt/4,
#                                 n_snapshot = 1,ode_step = 2,
#                                 time_evol = True)
#             output_quater2 = model(data,task_dt = args.task_dt/4,
#                                 n_snapshot = 1,ode_step = 2,
#                                 time_evol = True)
#             output_quater3 = model(data,task_dt = args.task_dt/4,
#                                 n_snapshot = 1,ode_step = 3,
#                                 time_evol = True)
#             output_quater4 = model(data,task_dt = args.task_dt/4,
#                                 n_snapshot = 1,ode_step = 4,
#                                 time_evol = True)
#             for i in range (target.shape[0]):
#                 RFNE_quater1 = torch.norm(output_quater1[i,0,...]-target[i,1,...])/torch.norm(target[i,1,...])
#                 RFNE_quater2 = torch.norm(output_quater2[i,0,...]-target[i,2,...])/torch.norm(target[i,2,...])
#                 RFNE_quater3 = torch.norm(output_quater3[i,0,...]-target[i,3,...])/torch.norm(target[i,3,...])
#                 RFNE_quater4 = torch.norm(output_quater4[i,0,...]-target[i,4,...])/torch.norm(target[i,4,...])
                
#                 list_quater1.append(RFNE_quater1)
#                 list_quater2.append(RFNE_quater2)
#                 list_quater3.append(RFNE_quater3)
#                 list_quater4.append(RFNE_quater4)

#     avg_quater1 = torch.mean(torch.stack(list_quater1),dim=0).item()
#     avg_quater2 = torch.mean(torch.stack(list_quater2),dim=0).item()
#     avg_quater3 = torch.mean(torch.stack(list_quater3),dim=0).item()
#     avg_quater4 = torch.mean(torch.stack(list_quater4),dim=0).item()

    return avg_quater1,avg_quater2,avg_quater3,avg_quater4

parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--model', type =str ,default= 'PASR')
parser.add_argument('--data', type =str ,default= 'nskt_16k')
parser.add_argument('--loss_type', type =str ,default= 'L1')
parser.add_argument('--scale_factor', type = int, default= 4)
parser.add_argument('--timescale_factor', type = int, default= 4)
parser.add_argument('--task_dt',type =float, default= 4)
parser.add_argument('--ode_step',type =int, default= 2)
parser.add_argument('--ode_method',type =str, default= "Euler")

parser.add_argument('--batch_size', type = int, default= 8)
parser.add_argument('--crop_size', type = int, default= 128, help= 'should be same as image dimension')
parser.add_argument('--epochs', type = int, default= 1)
parser.add_argument('--dtype', type = str, default= "float32")
parser.add_argument('--seed',type =int, default= 3407)


parser.add_argument('--n_snapshot',type =int, default= 20)
parser.add_argument('--down_method', type = str, default= "bicubic") # bicubic 
parser.add_argument('--upsampler', type = str, default= "pixelshuffle") # nearest+conv
parser.add_argument('--noise_ratio', type = float, default= 0.0)
parser.add_argument('--lr', type = float, default= 1e-4)
parser.add_argument('--lamb', type = float, default= 0.3)
parser.add_argument('--data_path',type = str,default = "../dataset/nskt16000_1024")
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
    mean = [0.1429] 
    std = [8.3615]
    model_list = {"PASR": PASR(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP":PASR_MLP(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
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
    checkpoint = torch.load("results/"+savedpath +".pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    epoch = checkpoint["epoch"]

    val_his = checkpoint["val_sum"]
    train_his = checkpoint["train_sum"]
    val_x_interpolate = checkpoint["val_x1"] 
    val_t_interpolate = checkpoint["val_t1"]
    val_x_extrapolate = checkpoint["val_x2"]
    val_t_extrapolate = checkpoint["val_t2"]

    fig,ax = plt.subplots(3,2,figsize = (12,12))
    fig.suptitle("train history, best at " + str(epoch))
    x = np.arange(0,len(train_his))
    ax[0,0].plot(x,train_his)
    ax[0,0].set_yscale("log")
    ax[0,0].set_title("train")
    ax[0,1].plot(x,val_his)
    ax[0,1].set_yscale("log")
    ax[0,1].set_title("val")
    ax[1,0].plot(x,val_x_interpolate)
    ax[1,0].set_yscale("log")
    ax[1,0].set_title("val_x_interpolate (dynamics = False)")
    ax[1,1].plot(x,val_t_interpolate)
    ax[1,1].set_yscale("log")
    ax[1,1].set_title("val_t_interpolate (dynamics = True)")    
    ax[2,0].plot(x,val_x_extrapolate)
    ax[2,0].set_yscale("log")
    ax[2,0].set_title("val_x_extrapolate (dynamics = False)")
    ax[2,1].plot(x,val_t_extrapolate)
    ax[2,1].set_yscale("log")
    ax[2,1].set_title("val_t_extrapolate (dynamics = True)")  
    fig.savefig("results/" + savedpath + ".png",dpi = 600)



    err1_interpolate,err2_interpolate = test_RFNE(model,test1_loader)
    err1_extrapolate,err2_extrapolate = test_RFNE(model,test2_loader)
    visualize(model,test1_loader,savedname = "interpolate_"+savedpath+".png")
    visualize(model,test2_loader,savedname = "extrapolate_"+savedpath+".png")
    # err1_interpolate_q,err2_interpolate_q,err3_interpolate_q,err4_interpolate_q = test_RFNE_quater(model,val1_loader)
    # err1_extrapolate_q,err2_extrapolate_q,err3_extrapolate_q,err4_extrapolate_q = test_RFNE_quater(model,test1_loader)
    with open("results/RFNE.txt","a") as f:
        print("============================================================",file =f)
        print(savedpath,file = f)
        print("RFNE_interpolate --- test t error: %.5f %%, test x error: %.5f %%" % (err1_interpolate*100.0, err2_interpolate*100.0),file =f )
        print("RFNE_extrapolate --- test t error: %.5f %%, test x error: %.5f %%" % (err1_extrapolate*100.0, err2_extrapolate*100.0),file =f )
        print("============================================================",file =f)
    logging.info("evaluation complete.")
    # print("RFNE_quater_interpolate --- test1 error: %.5f %%, test2 error: %.5f %%" % (err1_interpolate_q*100.0, err2_interpolate_q*100.0))
    # print("RFNE_quater_interpolate --- test3 error: %.5f %%, test4 error: %.5f %%" % (err3_interpolate_q*100.0, err4_interpolate_q*100.0))
    # print("RFNE_quater_extrapolate --- test1 error: %.5f %%, test2 error: %.5f %%" % (err1_extrapolate_q*100.0, err2_extrapolate_q*100.0))
    # print("RFNE_quater_extrapolate --- test3 error: %.5f %%, test4 error: %.5f %%" % (err3_extrapolate_q*100.0, err4_extrapolate_q*100.0))
