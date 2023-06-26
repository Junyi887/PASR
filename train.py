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
def validation(args,model, val1_loader,val2_loader,device):
    if args.loss_type =='L1':
        criterion_Data = nn.L1Loss().to(device)
    elif args.loss_type =='L2':
        criterion_Data = nn.MSELoss().to(device)

    target_loss1 = 0 
    input_loss1 = 0
    for batch in val1_loader: # better be the val loader, need to modify datasets, but we are good for now.
        with torch.no_grad():
            input, target = batch[0].float().to(device), batch[1].float().to(device)
            model.eval()
            out_t = model(input,task_dt = args.task_dt,n_snapshot = args.n_snapshot,ode_step = args.ode_step,time_evol = True) 
            out_x = model(input,task_dt = args.task_dt,n_snapshot = 1,ode_step = args.ode_step,time_evol = False) 
            target_loss = criterion_Data(out_t[:,-1,...], target[:,-1,...]) # Experiment change to criterion 1
            input_loss = criterion_Data(out_x[:,0,...], target[:,0,...]) # Experiment change to criterion 1
#            inerpolation_loss = criterion_Data(out[:,1,...], target[:,1,...]) # Experiment change to criterion 1
            target_loss1 += target_loss.item()
            input_loss1 += input_loss.item()

    target_loss2 = 0 
    input_loss2 = 0
    
    for batch in val2_loader: # better be the val loader, need to modify datasets, but we are good for now.
        with torch.no_grad():
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.eval()
            out_t = model(inputs,task_dt = args.task_dt,n_snapshot = args.n_snapshot,ode_step = args.ode_step,time_evol = True) 
            out_x = model(inputs,task_dt = args.task_dt,n_snapshot = 1,ode_step = args.ode_step,time_evol = False) 
            target_loss = criterion_Data(out_t[:,-1,...], target[:,-1,...]) # Experiment change to criterion 1
            input_loss = criterion_Data(out_x[:,0,...], target[:,0,...]) # Experiment change to criterion 1
#            inerpolation_loss = criterion_Data(out[:,1,...], target[:,1,...]) # Experiment change to criterion 1
            target_loss2 += target_loss.item()
            input_loss2 += input_loss.item()

    return input_loss1/len(val1_loader), target_loss1/len(val1_loader), input_loss2/len(val2_loader), target_loss2/len(val2_loader)

def train(args,model, trainloader, val1_loader,val2_loader, optimizer,device,savedpath):
    lamb = args.lamb
    val_list = []
    val_list_x1 = []
    val_list_t1 = []
    val_list_x2 = []
    val_list_t2 = []
    train_list = []
    best_loss_val = 1e9
    if args.loss_type =='L1':
        criterion_Data = nn.L1Loss().to(device)
    elif args.loss_type =='L2':
        criterion_Data = nn.MSELoss().to(device)

    for epoch in range(args.epochs):
        avg_loss = 0
        avg_val = 0
        for iteration, batch in enumerate(tqdm(trainloader)):
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.train()
            optimizer.zero_grad()
            out_x = model(inputs,task_dt = args.task_dt,n_snapshot = 1,ode_step = args.ode_step,time_evol = False)
            out_t = model(inputs,task_dt = args.task_dt,n_snapshot = args.n_snapshot,ode_step = args.ode_step,time_evol = True)

            loss_t = criterion_Data(out_t[:,-1,...], target[:,-1,...]) # Experiment change to criterion 1
            loss_x = criterion_Data(out_x[:,0,...], target[:,0,...])
            loss = loss_t + lamb*loss_x
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        val_x1,val_t1,val_x2,val_t2 = validation(args,model, val1_loader,val2_loader,device)
        avg_val = (val_x1 + val_x2)/2 + lamb*(val_t1 + val_t2)/2
        val_list.append(avg_val)
        val_list_x1.append(val_x1)
        val_list_t1.append(val_t1)
        val_list_x2.append(val_x2)
        val_list_t2.append(val_t2)
        train_list.append(avg_loss/len(trainloader))
        print("Epoch: {} | train loss: {} | val loss: {} | val_x1: {} | val_t1: {} | val_x2: {} | val_t2: {}".format(epoch, avg_loss/len(trainloader), avg_val, val_x1, val_t1, val_x2, val_t2))
        if avg_val < best_loss_val:
            best_loss_val = avg_val
            best_model = model
            torch.save({
            'epoch': epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_sum': np.array(val_list),
            'train_sum': np.array(train_list),
            'val_x1': np.array(val_list_x1),
            'val_t1': np.array(val_list_t1),
            'val_x2': np.array(val_list_x2),
            'val_t2': np.array(val_list_t2),
            },"results/"+savedpath + ".pt" ) # remember to change name for each experiment
        # validate 
    return 0



parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--data', type =str ,default= 'NSKT')
parser.add_argument('--loss_type', type =str ,default= 'L1')
parser.add_argument('--scale_factor', type = int, default= 4)
parser.add_argument('--timescale_factor', type = int, default= 4)
parser.add_argument('--batch_size', type = int, default= 1)
parser.add_argument('--crop_size', type = int, default= 512, help= 'should be same as image dimension')
parser.add_argument('--epochs', type = int, default= 1)
parser.add_argument('--dtype', type = str, default= "float32")
parser.add_argument('--seed',type =int, default= 3407)
parser.add_argument('--ode_step',type =int, default= 3)
parser.add_argument('--ode_method',type =str, default= "Euler")
parser.add_argument('--task_dt',type =float, default= 1)
parser.add_argument('--n_snapshot',type =int, default= 1)
parser.add_argument('--down_method', type = str, default= "uniform") # bicubic 
parser.add_argument('--upsampler', type = str, default= "pixelshuffle") # nearest+conv
parser.add_argument('--noise_ratio', type = float, default= 0.0)
parser.add_argument('--lr', type = float, default= 1e-4)
parser.add_argument('--lamb', type = float, default= 1.0)
parser.add_argument('--data_path',type = str,default = "../dataset/nskt16000_1024")
args = parser.parse_args()
print(args)

data_dx = 2*np.pi/2048
########### loaddata ############




if __name__ == "__main__":
    if args.dtype =="float32":
        data_type = torch.float32
    else: 
        data_type = torch.float64
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_default_dtype(data_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader,val1_loader,val2_loader,_,_ = getData(upscale_factor = args.scale_factor, timescale_factor= args.timescale_factor,batch_size = args.batch_size, crop_size = args.crop_size,data_path = args.data_path)
    mean = [0.1429] 
    std = [8.3615]
    model = PASR(upscale=args.scale_factor, in_chans=1, img_size=args.crop_size, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std).to(device,dtype=data_type)
    model = torch.nn.DataParallel(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    savedpath = str("PASR_" + str(args.ode_step) + 
                "_crop_size_" + str(args.crop_size) +
                "_ode_step_" + str(args.ode_step) +
                "ode_method_" + str(args.ode_method) +
                "_task_dt_" +  str(args.task_dt) + 
                "_num_snapshots_" + str(args.n_snapshot) +
                "_upscale_factor_" + str(args.scale_factor) +
                "_timescale_factor_" + str(args.timescale_factor) +
                "_loss_type_" + str(args.loss_type)
                ) 
    a = train(args,model, trainloader, val1_loader,val2_loader, optimizer, device, savedpath)