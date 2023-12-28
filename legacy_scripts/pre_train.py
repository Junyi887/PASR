import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import matplotlib.pyplot as plt
# 0.005
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
from legacy_scripts.data_loader_nersc import getData
import logging
import argparse
 
import neptune 
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import random
ID = random.randint(0, 10000)
run = neptune.init_run(
    project="junyiICSI/PASR",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGIxYjI4YS0yNDljLTQwOWMtOWY4YS0wOGNhM2Q5Y2RlYzQifQ==",
    tags = [str(ID)],
    )  # your credentials
# Replace the final print statement
def psnr(true, pred):
    mse = torch.mean((true - pred) ** 2)
    if mse == 0:
        return float(9999)
    max_value = torch.max(true)
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    if psnr.isnan() or psnr.isinf():
        return float(0)
    return psnr
    



def validation(args,model, val1_loader,val2_loader,device):
    if args.loss_type =='L1':
        criterion_Data = nn.L1Loss().to(device)
    elif args.loss_type =='L2':
        criterion_Data = nn.MSELoss().to(device)
    target_loss1 = 0 
    input_loss1 = 0
    RFNE1_loss = 0
    psnr1_loss = 0
    for batch in val1_loader: # better be the val loader, need to modify datasets, but we are good for now.
        with torch.no_grad():
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.eval()
            out_x = model(inputs,task_dt = args.task_dt,n_snapshots = 1,ode_step = args.ode_step,time_evol = False) 
            input_loss = criterion_Data(out_x[:,0,...], target[:,0,...]) # Experiment change to criterion 1
            out_t = model(inputs,task_dt = args.task_dt,n_snapshots = args.n_snapshots,ode_step = args.ode_step,time_evol = True) 
            loss_t = criterion_Data(out_t, target[:,1:,...])
            RFNE_t = torch.norm(out_t-target[:,1:,...],p=2,dim = (3,4))/torch.norm(target[:,1:,...],p=2,dim = (3,4))
            target_loss1 += loss_t.item() 
            input_loss1 += input_loss.item()
            RFNE1_loss += RFNE_t.mean().item()
            psnr1_loss += psnr(out_t, target[:,1:,...]).item()
    result_loader1 = [input_loss1/len(val1_loader), target_loss1/len(val1_loader), RFNE1_loss /len(val1_loader),psnr1_loss/len(val1_loader)]
    target_loss2 = 0 
    input_loss2 = 0
    RFNE2_loss = 0
    psnr2_loss = 0
    for batch in val2_loader: # better be the val loader, need to modify datasets, but we are good for now.
        with torch.no_grad():
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.eval()
            out_x = model(inputs,task_dt = args.task_dt,n_snapshots = 1,ode_step = args.ode_step,time_evol = False) 
            input_loss = criterion_Data(out_x[:,0,...], target[:,0,...]) # Experiment change to criterion 1
            out_t = model(inputs,task_dt = args.task_dt,n_snapshots = args.n_snapshots,ode_step = args.ode_step,time_evol = True) 
            loss_t = criterion_Data(out_t, target[:,1:,...])
            RFNE_t = torch.norm(out_t-target[:,1:,...],p=2,dim = (3,4))/torch.norm(target[:,1:,...],p=2,dim = (3,4))
            target_loss2 += loss_t.item() 
            input_loss2 += input_loss.item()
            RFNE2_loss += RFNE_t.mean().item()
            psnr2_loss += psnr(out_t, target[:,1:,...]).item()
    result_loader2 = [input_loss2/len(val2_loader), target_loss2/len(val2_loader), RFNE2_loss /len(val2_loader),psnr2_loss/len(val2_loader)]

    return result_loader1,result_loader2

def train(args,model, trainloader, val1_loader,val2_loader, optimizer,device,savedpath):
    lamb = args.lamb
    best_epoch = 0
    val_list = []
    val_list_x1 = []
    val_list_t1 = []
    val_list_x2 = []
    val_list_t2 = []
    train_list = []
    train_list_x = []
    train_list_t = []
    fd_solver = ConvFD(kernel_size=5).to(device)
    if args.scheduler == 'StepLR':
        scheduler = StepLR(optimizer, args.lr_step, gamma=args.gamma)
    elif args.scheduler == "Exp":
        scheduler = ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=args.gamma, patience=args.patience, min_lr=5e-6,threshold=0.005)
    best_loss_val = 1e9
    if args.loss_type =='L1':
        criterion_Data = nn.L1Loss().to(device)
    elif args.loss_type =='L2':
        criterion_Data = nn.MSELoss().to(device)
    criterion2 = nn.MSELoss().to(device)
    for epoch in range(args.epochs):
        avg_loss = 0
        avg_val = 0
        target_loss = 0
        input_loss = 0
        if args.scheduler == 'plateau':
            lr = optimizer.param_groups[0]["lr"]
            run['train/lr'].log(lr)
        else:
            run['train/lr'].log(scheduler.get_last_lr())
        for iteration, batch in enumerate(tqdm(trainloader)):
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.train()
            optimizer.zero_grad()
            out_x = model(inputs,task_dt = args.task_dt,n_snapshots = 1,ode_step = args.ode_step,time_evol = False)
            loss_x = criterion_Data(out_x[:,0,...], target[:,0,...])
            out_t = model(inputs,task_dt = args.task_dt,n_snapshots = args.n_snapshots,ode_step = args.ode_step,time_evol = True)
            loss_t = criterion_Data(out_t,target[:,1:,:,:,:])
            loss_t = torch.norm(out_t,target[:,1:,:,:,:],p=1,dim = (-1,-2))
            loss_t = loss_t.mean(dim = (0,2))
            div = fd_solver.get_div_loss(out_t)
            phy_loss = criterion2(div,torch.zeros_like(div).to(device)) # DO NOT CHANGE THIS ONE. Phy loss has to be L2 norm 
            if args.physics == "True":
                loss_t += args.lamb_p*phy_loss
            target_loss += loss_t.item() 
            input_loss += loss_x.item()
            loss = loss_t + lamb*loss_x
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        result_val1,result_val2 = validation(args,model, val1_loader,val2_loader,device)
        if args.scheduler == "plateau":
            scheduler.step(result_val1[1])
        else: 
            scheduler.step()

        avg_val = result_val1[1] + lamb*result_val1[0]
        val_list_x2.append(result_val2[2])
        run['train/train_loss'].log(avg_loss / len(trainloader))
        run['val/val_loss'].log(avg_val)
        run['train/train_loss_x'].log(input_loss / len(trainloader))
        run['train/train_loss_t'].log(target_loss / len(trainloader))
        run['val/val_loss_x1'].log(result_val1[0])
        run['val/val_loss_t1'].log(result_val1[1])
        run['test/test_loss_x1'].log(result_val2[0])
        run['test/test_loss_t1'].log(result_val2[1])
        run['val/RFNE'].log(result_val1[2])
        run['val/PSNR'].log(result_val1[3])
        run['test/RFNE'].log(result_val2[2])
        run['test/PSNR'].log(result_val2[3])
        run['train/div'].log(phy_loss.item())
        logging.info("Epoch: {} | train loss: {} | val loss: {} | val_x1: {} | val_t1: {} | val_x2: {} | val_t2: {}".format(epoch, avg_loss/len(trainloader), avg_val, result_val1[0], result_val1[1], result_val2[0],result_val2[1]))
        if avg_val < best_loss_val:
            best_loss_val = avg_val
            best_model = model
            best_epoch = epoch
            torch.save({
            'epoch': epoch,
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": vars(args)
            },"results/"+savedpath + ".pt" ) # remember to change name for each experiment
        # validate 
    return min(val_list_x2),best_epoch



parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--data', type =str ,default= 'Decay_turb')
parser.add_argument('--data_path',type = str,default = "../Decay_Turbulence")
parser.add_argument('--loss_type', type =str ,default= 'L1')
## data processing arugments
parser.add_argument('--in_channels',type = int, default= 1)
parser.add_argument('--batch_size', type = int, default= 8)
parser.add_argument('--crop_size', type = int, default= 256, help= 'should be same as image dimension')
parser.add_argument('--scale_factor', type = int, default= 4)
parser.add_argument('--timescale_factor', type = int, default= 1)
parser.add_argument('--n_snapshots',type =int, default= 20)
parser.add_argument('--down_method', type = str, default= "bicubic")
parser.add_argument('--noise_ratio', type = float, default= 0.0)
## model parameters 
parser.add_argument('--model', type =str ,default= 'PASR_MLP_small')
parser.add_argument('--upsampler', type = str, default= "pixelshuffle") # nearest+conv
parser.add_argument('--time_update',type =str, default= 'NODE')
parser.add_argument('--ode_step',type =int, default= 2)
parser.add_argument('--ode_method',type =str, default= "Euler")
parser.add_argument('--ode_kernel',type=int, default= 3)
parser.add_argument('--ode_padding',type=int, default= 1)
parser.add_argument('--ode_layer',type=int, default= 4)
parser.add_argument('--gating_layers',type =int, default= 3)
parser.add_argument('--gating_method',type =str, default= 'leaky')
parser.add_argument('--task_dt',type =float, default= 4)
## training (optimization) parameters
parser.add_argument('--epochs', type = int, default= 3)
parser.add_argument('--lr', type = float, default= 1e-4)
parser.add_argument('--lamb', type = float, default= 0.3)
parser.add_argument('--lamb_p', type = float, default= 1)
parser.add_argument('--dtype', type = str, default= "float32")
parser.add_argument('--seed',type =int, default= 3407)
parser.add_argument('--normalization',type =str, default= 'False')
parser.add_argument('--physics',type =str, default= 'False')
parser.add_argument('--gamma',type =float, default= 0.95)
parser.add_argument('--lr_step',type =int, default= 80)
parser.add_argument('--patience',type =int, default= 15)
parser.add_argument('--scheduler',type =str, default= 'StepLR')
args = parser.parse_args()
logging.info(args)

data_dx = 1/128
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
    logging.info(device)
    trainloader,val1_loader,val2_loader,_,_ = getData(upscale_factor = args.scale_factor, 
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size = args.batch_size, 
                                                      crop_size = args.crop_size,
                                                      data_path = args.data_path,
                                                      num_snapshots = args.n_snapshots,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=args.in_channels,)
    if args.normalization == "True":
        stats_loader = DataInfoLoader(args.data_path)
        mean, std = stats_loader.get_mean_std()
        min,max = stats_loader.get_min_max()
        if args.in_channels==1:
            mean,std = mean[0].tolist(),std[0].tolist()
            min,max = min[0].tolist(),max[0].tolist()
        elif args.in_channels==3:
            mean,std = mean.tolist(),std.tolist()
            min,max = min.tolist(),max.tolist()
        elif args.in_channels==2:
            mean,std = mean[1:].tolist(),std[1:].tolist()
            min,max = min[1:].tolist(),max[1:].tolist()
    else:
        mean, std = [0], [1]
        mean, std = mean * args.in_channels, std * args.in_channels


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
            "PASR_G":PASR_G(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,gating_layers=args.gating_layers,gating_method=args.gating_method),
            "PASR_MLP_small":PASR_MLP(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std),
            "PASR_G_small":PASR_G(upscale=args.scale_factor, in_chans=args.in_channels, img_size=image, window_size=8, depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler=args.upsampler, resi_conv='1conv',mean=mean,std=std,gating_layers=args.gating_layers,gating_method=args.gating_method),
    }
    model = torch.nn.DataParallel(model_list[args.model]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    savedpath = str(str(args.model) +
                "_data_" + str(args.data) + "_"+  str(ID)
                ) 

    run["config"] = vars(args)   
    min_RFNE,best_epoch = train(args,model, trainloader, val1_loader,val2_loader, optimizer, device, savedpath)
    run["metric/min_RFNE"].log(min_RFNE)
    run['metric/best_epoch'].log(best_epoch)
    run.stop()
    logging.info("Training complete.")
