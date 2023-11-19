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

# working on Negotiation 
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_train_history(epoch,train_his,val_his,val_x_interpolate,val_t_interpolate,val_x_extrapolate,val_t_extrapolate):
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
    return True

def plot_pixel_wise_visualize(model,test1_loader,location =100,savedname = 'test.png'):
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            if batch_idx == 4:
                data,target = data.float().to(device) , target.float().to(device)
                output_quater1 = model(data,task_dt = args.task_dt,
                                    n_snapshot = 4*args.n_snapshot,ode_step = args.ode_step,
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
            #logging.info("visualization complete.")
    return True

def _test_RFNE(model,test1_loader):
    list_t = []
    list_x = []
    with torch.no_grad():
        model.eval()
        for batch_idx,(data,target) in enumerate(test1_loader):
            data,target = data.to(device).float() , target.to(device).float()
            output_t = model(data,task_dt = args.task_dt,
                                    n_snapshot =args.n_snapshot,ode_step = args.ode_step,
                                    time_evol = True)
            output_x = model(data,task_dt = args.task_dt,
                                n_snapshot = 1,ode_step = args.ode_step,
                                time_evol = False)
            for i in range (target.shape[0]):
                for j in range (output_t.shape[1]):
                    RFNE_t = torch.norm(output_t[i,j,...]-target[i,j+1,...])/torch.norm(target[i,j+1,...])
                    list_t.append(RFNE_t)
                RFNE_x = torch.norm(output_x[i,0,...]-target[i,0,...])/torch.norm(target[i,0,...])
                list_x.append(RFNE_x)
    avg_t = torch.mean(torch.stack(list_t),dim=0).item()
    avg_x = torch.mean(torch.stack(list_x),dim=0).item()
    logging.info("Test RFNE complete.")
    return avg_t,avg_x

def _test_RFNE_n_length(model,test1_loader):
    list_t = []
    list_n_snap = []
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            data,target = data.to(device).float() , target.to(device).float()
            if batch_idx == 4:
                for n_snapshot in range (1,args.n_snapshot*3):
                    list_t = []
                    output_t = model(data,task_dt = args.task_dt,
                                    n_snapshot =n_snapshot,ode_step = args.ode_step,
                                    time_evol = True)
                    i = 1
                    for j in range (output_t.shape[1]):
                        RFNE_t = torch.norm(output_t[i,j,...]-target[i,j+1,...])/torch.norm(target[i,j+1,...])
                        list_t.append(RFNE_t)
                    avg_t = torch.mean(torch.stack(list_t),dim=0).item()
                    list_n_snap.append(avg_t)
                logging.info("Test RFNE prediction length complete.") 
                return list_n_snap     
    return None

def _test_RFNE_no_pred(model,test1_loader):
    list_t = []
    list_n_snap = []
    with torch.no_grad():
        for batch_idx,(data,target) in enumerate(test1_loader):
            data,target = data.to(device).float() , target.to(device).float()
            if batch_idx == 4:
                for n_snapshot in range (1,args.n_snapshot*3):
                    list_t = []
                    output_t = model(data,task_dt = 0.0,
                                    n_snapshot = n_snapshot,ode_step = args.ode_step,
                                    time_evol = False)
                    i = 1
                    for j in range (output_t.shape[1]):
                        RFNE_t = torch.norm(output_t[i,0,...]-target[i,j+1,...])/torch.norm(target[i,j+1,...])
                        list_t.append(RFNE_t)
                    avg_t = torch.mean(torch.stack(list_t),dim=0).item()
                    list_n_snap.append(avg_t)
                logging.info("No Dynamics RFNE complete.") 
                return list_n_snap     
    return None

def _test_RFNE_many(model,test1_loader):
    list_t = []
    list_tt = []
    with torch.no_grad():
        for num_snap in range (1,args.n_snapshot*4):
            list_t = []
            for batch_idx,(data,target) in enumerate(test1_loader):
                data,target = data.to(device).float() , target.to(device).float()
                output_t = model(data,task_dt = args.task_dt/4,
                                    n_snapshot = num_snap ,ode_step = args.ode_step//4,
                                    time_evol = True)
                for i in range (target.shape[0]):
                    for j in range (output_t.shape[1]):
                        RFNE_t = torch.norm(output_t[i,j,...]-target[i,j+1,...])/torch.norm(target[i,j+1,...])
                        list_t.append(RFNE_t)
            avg_t = torch.mean(torch.stack(list_t),dim=0).item()
            list_tt.append(list_tt)
    logging.info("Test RFNE many complete.")
    return list_tt

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
parser.add_argument('--data_path',default = None)
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
    _,_,_,test1_loader,test2_loader = getData(upscale_factor = args.scale_factor, timescale_factor= args.timescale_factor,batch_size = args.batch_size, crop_size = args.crop_size,data_path = args.data_path,num_snapshots = args.n_snapshot*3,data_name = args.data)
    mean = [0] 
    std = [1]
    model_list = {"PASR": PASR(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP":PASR_MLP(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP_G":PASR_MLP_G(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP_small":PASR_MLP(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP_G_small":PASR_MLP_G(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP_G_aug":PASR_MLP_G_aug(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),
            "PASR_MLP_G_aug_small":PASR_MLP_G_aug(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type),

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
    if args.model_path ==None:
        checkpoint = torch.load("results/"+savedpath +".pt")
    else:
        checkpoint = torch.load(args.model_path)

    model.load_state_dict(checkpoint["model_state_dict"])
    # epoch = checkpoint["epoch"]

    # val_his = checkpoint["val_sum"]
    # train_his = checkpoint["train_sum"]
    # val_x_interpolate = checkpoint["val_x1"] 
    # val_t_interpolate = checkpoint["val_t1"]
    # val_x_extrapolate = checkpoint["val_x2"]
    # val_t_extrapolate = checkpoint["val_t2"]






    def plot_pixel_evolution_compare(model,test1_loader,test2_loader):
        list_t1 = _test_RFNE_n_length(model,test1_loader)
        list_t2 = _test_RFNE_n_length(model,test2_loader)
        list_t11 = _test_RFNE_no_pred(model,test1_loader)
        list_t22 = _test_RFNE_no_pred(model,test2_loader)
        fig,ax = plt.subplots(1,2,figsize = (12,6))
        ax[0].scatter(range(len(list_t1)),list_t1,label='dynamics',marker = "*")
        ax[0].scatter(range(len(list_t11)),list_t11,label='no dynamics reference',marker = "+")
        ax[0].legend()
        ax[0].set_title("test1")
        ax[0].set_xlabel("n_snapshot")
        ax[0].set_ylabel("RFNE")
        ax[1].scatter(range(len(list_t2)),list_t2,label='dynamics',marker = "*")
        ax[1].scatter(range(len(list_t22)),list_t22,label='no dynamics reference',marker = "+")
        ax[1].legend()
        ax[1].set_title("test2")
        ax[1].set_xlabel("n_snapshot")
        ax[1].set_ylabel("RFNE")
        fig.savefig("results/" + "n_snap_" +savedpath + ".png",dpi = 600)
        return True
    

    # err1_interpolate_q,err2_interpolate_q,err3_interpolate_q,err4_interpolate_q = test_RFNE_quater(model,val1_loader)
    # err1_extrapolate_q,err2_extrapolate_q,err3_extrapolate_q,err4_extrapolate_q = test_RFNE_quater(model,test1_loader)


    def getRFNE(model,test1_loader,test2_loader):
        err1_interpolate,err2_interpolate = _test_RFNE(model,test1_loader)
        err1_extrapolate,err2_extrapolate = _test_RFNE(model,test2_loader)
        with open("results/RFNE.txt","a") as f:
            print("============================================================",file =f)
            print(savedpath,file = f)
            print("RFNE_interpolate --- test t error: %.5f %%, test x error: %.5f %%" % (err1_interpolate*100.0, err2_interpolate*100.0),file =f )
            print("RFNE_extrapolate --- test t error: %.5f %%, test x error: %.5f %%" % (err1_extrapolate*100.0, err2_extrapolate*100.0),file =f )
            
            print("============================================================",file =f)
        return True 


if getRFNE(model,test1_loader,test2_loader): logging.info("RFNE evaluation complete.")


    # plot_pixel_wise_visualize(model,test1_loader,savedname = "interpolate_"+savedpath+".png")
    # plot_pixel_wise_visualize(model,test2_loader,savedname = "extrapolate_"+savedpath+".png")

class Visualize():
    def __init__(self,args,model,test_loader,tasks):
        self.model = model
        self.test_loader = test_loader
        self.tasks = tasks
    def _get_single_batch_predictions(self,loader,idx):
        for batch_idx,(data,target) in enumerate(loader):
            if batch_idx == idx:
                data,target = data.float().to(device) , target.float().to(device)
                pred = model(data,task_dt = args.task_dt,
                                    n_snapshot = 4*args.n_snapshot,ode_step = args.ode_step,
                                    time_evol = True)
                target = target
                break
        return pred,target
    
    def _get_single_snapshot_predictions(self,loader,idx):
        for batch_idx,(data,target) in enumerate(loader):
            if batch_idx == idx:
                data,target = data.float().to(device) , target.float().to(device)
                pred = model(data,task_dt = args.task_dt,
                                    n_snapshot = 4*args.n_snapshot,ode_step = args.ode_step,
                                    time_evol = True)
                target = target
                break
        return pred,target
    
        # if "single_pixel" in self.tasks:
        #     new_pred_dic = {"single_pixel":(pred[0,:,0,loc_x,loc_y])}
        #     new_target_dic = {"single_pixel":(target[0,:,0,loc_x,loc_y])}
        #     pred_dic.update(new_pred_dic)
        #     target_dic.update(new_target_dic)
        # if "multi_pixel" in self.tasks:
        #     new_pred_dic = {"multi_pixel":(pred[0,:,0,:,:])}
        #     new_target_dic = {"multi_pixel":(target[0,:,0,:,:])}
        #     pred_dic.update(new_pred_dic)
        #     target_dic.update(new_target_dic)
        return 