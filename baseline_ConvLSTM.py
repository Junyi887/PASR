'''PhySR for RB equation'''

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau 
from torch.nn.utils import weight_norm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.utils.checkpoint import checkpoint
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import os
import argparse
import math
import logging
import torch.nn.init as init
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
import neptune
import time
# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import random
ID = random.randint(0, 10000)

run = neptune.init_run(
    project="junyiICSI/PASR",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NGIxYjI4YS0yNDljLTQwOWMtOWY4YS0wOGNhM2Q5Y2RlYzQifQ==",
    tags = [str(ID)],
    # mode = "debug"
    )  # your credentials
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'



lapl_op = [[[[    0,   0, -1/12,   0,     0],
             [    0,   0,   4/3,   0,     0],
             [-1/12, 4/3,    -5, 4/3, -1/12],
             [    0,   0,   4/3,   0,     0],
             [    0,   0, -1/12,   0,     0]]]]

par_y = [[[[    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0],
           [1/12, -8/12,  0,  8/12, -1/12],
           [    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0]]]]


par_x = [[[[    0,   0,   1/12,   0,     0],
           [    0,   0,   -8/12,   0,     0],
           [    0,   0,   0,   0,     0],
           [    0,   0,   8/12,   0,     0],
           [    0,   0,   -1/12,   0,     0]]]]



class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class Conv1dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv1dDerivative, self).__init__()

        self.resol = resol  # $\delta$*constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) / 2)
        self.filter = nn.Conv1d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)
        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol


class LossGenerator(nn.Module):
    '''Calculate the physics loss and the data loss'''

    def __init__(self, dt = (10.0/200), dx = (20.0/128)):
       
        super(LossGenerator, self).__init__()

        # spatial derivative operator
        self.laplace = Conv2dDerivative(
            DerFilter = lapl_op,
            resol = (dx**2),
            kernel_size = 5,
            name = 'laplace_operator').cuda()

        # forward/backward derivative operator 
        self.dt = Conv1dDerivative(
            DerFilter = [[[-1/2, 0, 1/2]]],
            resol = (dt),
            kernel_size = 3,
            name = 'partial_t').cuda() 

        self.fwd_dt = Conv1dDerivative(
            DerFilter = [[[-3/2, 2, -1/2]]],
            resol = (dt),
            kernel_size = 3,
            name = 'forward_partial_t').cuda() 
 
        self.bwd_dt = Conv1dDerivative(
            DerFilter = [[[1/2, -2, 3/2]]],
            resol = (dt),
            kernel_size = 3,
            name = 'backward_partial_t').cuda() 

        # FD kernels        
        self.dx = Conv2dDerivative(
            DerFilter = par_x,
            resol = dx,
            kernel_size = 5,
            name = 'dx_operator').cuda()

        self.dy = Conv2dDerivative(
            DerFilter = par_y,
            resol = dx,
            kernel_size = 5,
            name = 'dy_operator').cuda()        


    def get_div_loss(self, output):
        '''compute divergence loss'''
        # output: [t,b,c,h,w]
        # [p,T,u,v]

        # laplace u, [t-2,b,c,h,w]
        u = output[:, :, 1:2, :, :]
        len_t,len_b,len_c,len_h,len_w = u.shape 
        # [t,b,c,h,w] -> [t*b,c,h,w]
        u = u.reshape(len_t*len_b, len_c, len_h, len_w)
        u_x = self.dx(u)  
        # change batch to [t,b,c,h,w]
        u_x = u_x.reshape(len_t,len_b,len_c,len_h-4,len_w-4)

        # laplace v, [t-2,b,c,h,w]
        v = output[:, :, 2:3, :, :]
        len_t,len_b,len_c,len_h,len_w = v.shape 
        v = v.reshape(len_t*len_b, len_c, len_h, len_w)
        v_y = self.dy(v)  
        v_y = v_y.reshape(len_t,len_b,len_c,len_h-4,len_w-4)
        
        # div
        div = u_x + v_y

        return div


    def GetPhyLoss(self, output):
        '''Calculate the physical loss'''
        # output: [t,b,c,h,w]
        # [p,T,u,v]

        ############### spatial derivatives #################
        # u_x, u_y, v_x, v_y, u_xx, u_yy
        # laplace u
        u = output[:, :, 1, :, :]

        ############### temporal derivatives #################

        ############### spatial derivatives #################
        # laplace u, [t-2,b,c,h,w]
        u = output[:, :, 1:2, :, :]
        len_t,len_b,len_c,len_h,len_w = u.shape 
        # [t,b,c,h,w] -> [t*b,c,h,w]
        u = u.reshape(len_t*len_b, len_c, len_h, len_w)
        laplace_u = self.laplace(u)  
        # change batch to [t,b,c,h,w]
        laplace_u = laplace_u.reshape(len_t,len_b,len_c,len_h-4,len_w-4)

        # laplace v, [t-2,b,c,h,w]
        v = output[:, :, 2:3, :, :]
        len_t,len_b,len_c,len_h,len_w = v.shape 
        v = v.reshape(len_t*len_b, len_c, len_h, len_w)
        laplace_v = self.laplace(v)  
        laplace_v = laplace_v.reshape(len_t,len_b,len_c,len_h-4,len_w-4)

        ############### temporal derivatives #################
        # u_t, [t,b,c,h-4,w-4]
        u = output[:, :, 1:2, 2:-2, 2:-2]
        len_t,len_b,len_c,len_h,len_w = u.shape 
        u = u.permute(3,4,1,2,0) # [h,w,b,c,t]
        u = u.reshape(len_h*len_w*len_b, len_c, len_t) # [h*w*b,c,t]
        u_t = self.dt(u) # [h*w*b,c,t-2]
        u_t0 = self.fwd_dt(u[:,:,0:3])
        u_tn = self.bwd_dt(u[:,:,-3:])
        u_t = torch.cat((u_t0,u_t,u_tn), dim=2) # [h*w*b,c,t]
        u_t = u_t.reshape(len_h, len_w, len_b, len_c, len_t)
        u_t = u_t.permute(4,2,3,0,1)

        # v_t, [t,b,c,h-4,w-4]
        v = output[:, :, 2:3, 2:-2, 2:-2]
        len_t,len_b,len_c,len_h,len_w = v.shape 
        v = v.permute(3,4,1,2,0) # [h,w,b,c,t]
        v = v.reshape(len_h*len_w*len_b, len_c, len_t) # [h*w*b,c,t]
        v_t = self.dt(v)
        v_t0 = self.fwd_dt(v[:,:,0:3])
        v_tn = self.bwd_dt(v[:,:,-3:])
        v_t = torch.cat((v_t0, v_t, v_tn), dim=2)
        v_t = v_t.reshape(len_h, len_w, len_b, len_c, len_t)
        v_t = v_t.permute(4,2,3,0,1)

        ############### corresponding u & v ###################
        u = output[:, :, 1:2, 2:-2, 2:-2]  # [step, b, c, height(Y), width(X)]
        v = output[:, :, 2:3, 2:-2, 2:-2]  # [step, b, c, height(Y), width(X)]

        # make sure the dimensions consistent
        assert laplace_u.shape == u_t.shape
        assert u_t.shape == v_t.shape
        assert laplace_u.shape == u.shape
        assert laplace_v.shape == v.shape

        # 2D GS
        Du, Dv = 0.16, 0.08
        f, k = 0.06, 0.062
        f_u = Du*laplace_u - u*v**2 + f*(1-u) - u_t
        f_v = Dv*laplace_v + u*v**2 - (f+k)*v - v_t

        return f_u, f_v


def LossGen(output, truth, beta, loss_func):

    L1_loss = nn.L1Loss()
    MSE_loss = nn.MSELoss()

    # data loss d
    data_loss = L1_loss(output, truth)
    
    # ic loss
    ic_loss = 0

    # phy loss, output shape: [t,b,c,h,w]
    # output = torch.cat((output[:, :, :, :, -2:], output, output[:, :, :, :, 0:3]), dim=4)
    # output = torch.cat((output[:, :, :, -2:, :], output, output[:, :, :, 0:3, :]), dim=3)
    
    # divergence loss
    div = 0
    phy_loss = 0
    
    # f_u, f_v = loss_func.GetPhyLoss(output)
    #phy_loss = MSE_loss(f_u, torch.zeros_like(f_u).cuda()) + MSE_loss(
    #            f_v, torch.zeros_like(f_v).cuda())

    loss = data_loss + beta * phy_loss + 1.0 * ic_loss 

    return loss, data_loss, phy_loss 


def train(model, train_loader, val_loader, init_state, n_iters, lr, print_every, dt, dx, 
    beta, save_path, pretrain_flag=False):
    # train_loader: low resolution tensor
    # beta works on physics loss

    best_error = 1e2
    print_loss_total = 0
    train_loss_list, val_loss_list, val_error_list = [], [], []
    pretrain_save_path = save_path + 'pretrain.pt'
    model_save_path = save_path + 'checkpoint.pt'

    if pretrain_flag == True:
        model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, 
            save_dir=pretrain_save_path) 

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.998) 
    loss_function = LossGenerator(dt, dx)

    for epoch in range(n_iters):
        for idx, (lres, hres) in enumerate(train_loader):
            
            optimizer.zero_grad()

            lres, hres = lres.float().to(device), hres.float().to(device) # B C T H W
            # lres, hres = lres.permute(2,0,1,3,4), hres.permute(2,0,1,3,4) # (b,t,c,h,w) 
            outputs = model(lres, init_state)
            # compute loss 
            loss, data_loss, phy_loss = LossGen(outputs, hres, beta, loss_function)
            loss.backward(retain_graph=True)
            print_loss_total += loss.item()

            # gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

            optimizer.step()
            scheduler.step()

        if (epoch+1) % print_every == 0:
            # calculate the average training loss
            print_loss_mean = print_loss_total / (print_every*len(train_loader))
            train_loss_list.append(print_loss_mean)
            print_loss_total = 0

            # print the training loss
            print('Train loss (%d/%d %d%%): %.8f'  % (epoch+1, n_iters, 
                (epoch+1)/n_iters*100, print_loss_mean))

            # for print training loss (details)
            # print('Epoch %d: data loss(%.8f), phy loss(%.8f)' %(
            #     epoch+1, data_loss.item(), phy_loss.item()))
            # calculate the validation loss
            val_loss, val_error = validate(model, val_loader, init_state, loss_function, beta)
            run["train/val_loss"].log(val_loss)
            run["train/val_error"].log(val_error)
            run["train/train_loss"].log(print_loss_mean)
            run["train/lr"].log(scheduler.get_last_lr())
            val_loss_list.append(val_loss)
            val_error_list.append(val_error)

            # for print validation loss
            print('Epoch (%d/%d %d%%): val loss %.8f, val error %.8f'  % (epoch+1, n_iters, 
                (epoch+1)/n_iters*100, val_loss, val_error))
            print('')
            # save model
            if val_error < best_error:
                save_checkpoint(model, optimizer, scheduler, model_save_path)
                best_error = val_error
        if (epoch+1) % 100 == 0:
            pred_error = test(model, val2_loader, init_state, save_path, fig_save_path)
            run["train/test_error"].log(pred_error)
            print('Epoch (%d/%d %d%%): val loss %.8f, val error %.8f, test error %.8f'  % (epoch+1, n_iters, 
                (epoch+1)/n_iters*100, val_loss, val_error, pred_error))
    return train_loss_list, val_loss_list, val_error_list


def validate(model, val_loader, init_state, loss_function, beta):
    ''' evaluate the model performance '''
    val_loss = 0
    val_error = 0
    MSE_function = nn.MSELoss()

    for idx, (lres, hres) in enumerate(val_loader):

        lres, hres = lres.float().cuda(), hres.float().cuda()  
        # lres, hres = lres.permute(2,0,1,3,4), hres.permute(2,0,1,3,4) # (b,c,t,h,w) -> (t,b,c,h,w)

        outputs = model(lres, init_state)
 
        # calculate the loss
        loss,_,_ = LossGen(outputs, hres, beta, loss_function)
        val_loss += loss.item()
        
        # calculate the error
        error = torch.norm(hres-outputs.detach(),p=2,dim = (-1,-2)) / torch.norm(hres,p=2,dim = (-1,-2))
        val_error += error.mean().item()

    val_error = val_error / len(val_loader) 
    val_loss = val_loss / len(val_loader)

    return val_loss, val_error


def test(model, test_loader, init_state, save_path, fig_save_path):
    # load the well-trained model
    model_save_path = save_path + 'checkpoint.pt'
    model, _, _ = load_checkpoint(model, optimizer=None, scheduler=None, 
        save_dir=model_save_path) 

    MSE_function = nn.MSELoss()
    pred_error = 0

    for idx, (lres, hres) in enumerate(test_loader):

        lres, hres = lres.float().cuda(), hres.float().cuda()  
        outputs = model(lres, init_state)

        # calculate the error
        error = torch.norm(hres-outputs.detach(),p=2,dim = (-1,-2)) / torch.norm(hres,p=2,dim = (-1,-2))
        pred_error += error.mean().item()

    pred_error = pred_error/len(test_loader)

    return pred_error


def save_checkpoint(model, optimizer, scheduler, save_dir):
    '''save model and optimizer'''
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
        }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir):
    '''load model and optimizer'''
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    if (not optimizer is None):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training parameters')

    parser.add_argument('--data', type =str ,default= 'DT_sequenceLR')
    parser.add_argument('--data_path',type = str,default = "../decay_turb_lrsim")
    ## data processing arugments
    parser.add_argument('--in_channels',type = int, default= 3)
    parser.add_argument('--batch_size',type = int, default= 64)
    parser.add_argument('--scale_factor', type = int, default= 4)
    parser.add_argument('--timescale_factor', type = int, default= 4)
    parser.add_argument('--n_snapshots',type =int, default= 20)
    parser.add_argument('--down_method', type = str, default= "bicubic")
    parser.add_argument('--noise_ratio', type = float, default= 0.0)
    parser.add_argument('--normalization', type = str, default= "True")
    parser.add_argument('--normalization_method', type = str, default= "meanstd")
    parser.add_argument('--seed', type = int, default= 0)
    args = parser.parse_args()
    run["config"] = vars(args) 

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float32)
    # define the data file path 
    trainloader,val1_loader,val2_loader,_,_ = getData(upscale_factor = args.scale_factor, 
                                                      timescale_factor= args.timescale_factor,
                                                      batch_size = args.batch_size, 
                                                      crop_size = 256,
                                                      data_path = args.data_path,
                                                      num_snapshots = args.n_snapshots,
                                                      noise_ratio = args.noise_ratio,
                                                      data_name = args.data,
                                                      in_channels=args.in_channels,)
    # get mean and std

    stats_loader = DataInfoLoader(args.data_path+"/*/*.h5")
    def get_normalizer(args,stats_loader=stats_loader):
        if args.normalization == "True":
            mean, std = stats_loader.get_mean_std()
            min,max = stats_loader.get_min_max()
            if args.in_channels==1:
                mean,std = mean[0:1].tolist(),std[0:1].tolist()
                min,max = min[0:1].tolist(),max[0:1].tolist()
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
    shape_x,shape_y = stats_loader.get_shape()

    ######################### build model #############################
    # training parameters
    n_iters = 400 # 500 
    learning_rate = 1e-3
    print_every = 2   
    dt = 0.01
    dx = 1.0 / 64 
    steps = 21 # 40 
    effective_step = list(range(0, steps))
    
    beta = 0.0 # 0.025 # for physics loss        
    save_path = 'ConvLSTM_' + str(args.data) +str(ID) +"_"
    fig_save_path = 'ConvLSTM_'+ str(args.data)+str(ID)+"_"
    print('Super-Resolution for 2D RB equation...')

    model = PhySR(
        n_feats = 32,
        n_layers = [1, 2], # [n_convlstm, n_resblock]
        upscale_factor = [args.n_snapshots, 4], # [t_up, s_up]
        shift_mean_paras = [mean, std],  
        step = steps,
        in_channels=args.in_channels,
        effective_step = effective_step)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.nn.DataParallel(model).to(device)
    # define the initial states and initial output for model
    init_state = get_init_state(
        batch_size = [args.batch_size], 
        hidden_channels = [32], 
        output_size = [[shape_x//args.scale_factor, shape_y//args.scale_factor]], # 32, 32 
        mode = 'random')

    start = time.time()
    train_loss_list, val_loss_list, val_error_list = train(model, trainloader, val1_loader, 
        init_state, n_iters, learning_rate, print_every, dt, dx, beta, save_path)
    end = time.time()
    print('The training time is: ', (end - start))
    print('')

    np.save(save_path + 'train_loss', train_loss_list)
    np.save(save_path + 'val_loss', val_loss_list)
    np.save(save_path + 'val_error', val_error_list)

    ###################### model inference ###########################
    pred_error = test(model, val2_loader, init_state, save_path, fig_save_path)
    print('The predictive error is: ', pred_error)
    print('Test completed')

    # plot loss
    x_axis = np.arange(0, n_iters, print_every)
    plt.figure()
    plt.plot(x_axis, train_loss_list, label = 'train loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_save_path + 'train loss.png', dpi = 300)

    plt.figure()
    plt.plot(x_axis, val_loss_list, label = 'val loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_save_path + 'val loss.png', dpi = 300)

    plt.figure()
    plt.plot(x_axis, val_error_list, label = 'val error')
    plt.yscale('log')
    plt.legend()
    plt.savefig(fig_save_path + 'val error.png', dpi = 300)


