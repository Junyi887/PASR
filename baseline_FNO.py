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
from src.data_loader_nersc import getData
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
    for batch in val1_loader: 
        with torch.no_grad():
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.eval()
            out_t = model(inputs) 
            loss_t = criterion_Data(out_t, target)
            #B,C,T,H,W
            RFNE_t = torch.norm(out_t-target,p=2,dim = (-1,-2))/torch.norm(target,p=2,dim = (-1,-2))
            target_loss1 += loss_t.item() 
            input_loss1 += 0
            RFNE1_loss += RFNE_t.mean().item()
            psnr1_loss += psnr(out_t, target).item()
    result_loader1 = [input_loss1/len(val1_loader), target_loss1/len(val1_loader), RFNE1_loss /len(val1_loader),psnr1_loss/len(val1_loader)]
    target_loss2 = 0 
    input_loss2 = 0
    RFNE2_loss = 0
    psnr2_loss = 0
    for batch in val2_loader: # better be the val loader, need to modify datasets, but we are good for now.
        with torch.no_grad():
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.eval()
            out_t = model(inputs) 
            loss_t = criterion_Data(out_t, target)
            #B,C,T,H,W
            RFNE_t = torch.norm(out_t-target,p=2,dim = (-1,-2))/torch.norm(target,p=2,dim = (-1,-2))
            target_loss2 += loss_t.item() 
            input_loss2 += 0
            RFNE2_loss += RFNE_t.mean().item()
            psnr2_loss += psnr(out_t, target).item()
    result_loader2 = [input_loss2/len(val2_loader), target_loss2/len(val2_loader), RFNE2_loss /len(val2_loader),psnr2_loss/len(val2_loader)]
    return result_loader1,result_loader2

def train(args,model, trainloader, val1_loader,val2_loader, optimizer,device,savedpath):
    lamb = args.lamb
    best_epoch = 0
    val_list_x2 = []
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
        for iteration, batch in enumerate(trainloader):
            inputs, target = batch[0].float().to(device), batch[1].float().to(device)
            model.train()
            optimizer.zero_grad()
            out = model(inputs)
            loss = criterion_Data(out,target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        scheduler.step()
        result_val1,result_val2 = validation(args,model, val1_loader,val2_loader,device)
        avg_val = result_val1[1]
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
            "config": vars(args),
            },"results/"+savedpath + ".pt" ) # remember to change name for each experiment
        # validate 
    return min(val_list_x2),best_epoch



parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--data', type =str ,default= 'Decay_turb_FNO')
parser.add_argument('--data_path',type = str,default = "../Decay_Turbulence_small")
## data processing arugments
parser.add_argument('--scale_factor', type = int, default= 4)
parser.add_argument('--timescale_factor', type = int, default= 4)
parser.add_argument('--in_channels',type = int, default= 3)
parser.add_argument('--batch_size', type = int, default= 8)
parser.add_argument('--n_snapshots',type =int, default= 20)
parser.add_argument('--down_method', type = str, default= "bicubic") # bicubic 
parser.add_argument('--noise_ratio', type = float, default= 0.0)
parser.add_argument('--crop_size', type = int, default= 256)
## model parameters 
parser.add_argument('--model', type = str, default= "FNO")
parser.add_argument('--modes', type = int, default= 12)
parser.add_argument('--width', type = int, default= 16)
parser.add_argument('--hidden_dim', type = int, default= 64) # euler
## training (optimization) parameters
parser.add_argument('--epochs', type = int, default= 500)
parser.add_argument('--loss_type', type =str ,default= 'L2')
parser.add_argument('--dtype', type = str, default= "float32")
parser.add_argument('--seed',type =int, default= 3407)
parser.add_argument('--normalization',type =str, default= 'True')
parser.add_argument('--normalization_method',type =str, default= 'meanstd')
parser.add_argument('--physics',type =str, default= 'False')
parser.add_argument('--gamma',type =float, default= 0.5)
parser.add_argument('--lr_step',type =int, default= 100)
parser.add_argument('--patience',type =int, default= 15)
parser.add_argument('--scheduler',type =str, default= 'StepLR')
parser.add_argument('--lr', type = float, default= 1e-3)
parser.add_argument('--lamb', type = float, default= 0.3)
parser.add_argument('--lamb_p', type = float, default= 1)


args = parser.parse_args()
logging.info(args)

data_dx = 1/128



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
    if args.data =="Decay_turb_small" or args.data =="Decay_turb_FNO": 
        image = [128,128]
        target_shape = (args.batch_size,args.in_channels,args.n_snapshots+1,128,128)
        ALL_DATA_PATH = "/pscratch/sd/j/junyi012/Decay_Turbulence_small/*/*.h5"
    elif args.data =="rbc_small" or args.data =="rbc_FNO":
        image = [256,64]
        target_shape = (args.batch_size,args.in_channels,args.n_snapshots+1,256,64)
        ALL_DATA_PATH = "/pscratch/sd/j/junyi012/RBC_small/*/*.h5"
    elif args.data =="Burger2D_small" or args.data =="Burger2D_FNO":
        image = [128,128]
        target_shape = (args.batch_size,args.in_channels,args.n_snapshots+1,128,128)
        ALL_DATA_PATH = "../burger2D_10/*/*.h5"

    def get_normalizer(args):
        if args.normalization == "True":
            stats_loader = DataInfoLoader(args.data_path+"/*/*.h5")
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
            if args.normalization_method =="minmax":
                return min,max
            if args.normalization_method =="meanstd":
                return mean,std
        else:
            mean, std = [0], [1]
            mean, std = mean * args.in_channels, std * args.in_channels
            return mean,std
    mean,std = get_normalizer(args)

    layers = [64, 64, 64, 64, 64]
    modes1 = [8, 8, 8, 8]
    modes2 = [8, 8, 8, 8]
    modes3 = [8, 8, 8, 8]


    model = FNO3D(modes1, modes2, modes3,target_shape,width=args.width, fc_dim=args.hidden_dim,layers=layers,in_dim=args.in_channels, out_dim=args.in_channels, act='gelu', mean=mean,std=std).to(device)
    # model = torch.nn.DataParallel(model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    savedpath = str(str(args.model) +
                "_data_" + str(args.data) +"_"+ str(ID)
                ) 

    run["config"] = vars(args)   
    min_RFNE,best_epoch = train(args,model, trainloader, val1_loader,val2_loader, optimizer, device, savedpath)
    run["metric/min_RFNE"].log(min_RFNE)
    run['metric/best_epoch'].log(best_epoch)
    run.stop()
    logging.info("Training complete.")
