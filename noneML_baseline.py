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
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import CubicSpline
# trainloader,_,_,_,_ = getData(upscale_factor = 4, 
#                                 timescale_factor= 4,
#                                 batch_size = 1000, 
#                                 crop_size = 256,
#                                 data_path = "../dataset/nskt16000_1024",
#                                 num_snapshots = 4,
#                                 noise_ratio = 0)
# _,val1,_,_,_ = getData(upscale_factor = 4, 
#                                 timescale_factor= 4,
#                                 batch_size = 200, 
#                                 crop_size = 256,
#                                 data_path = "../dataset/nskt16000_1024",
#                                 num_snapshots = 4,
#                                 noise_ratio = 0)

# for i,(data,target) in enumerate(trainloader):
#     print(data.shape)
#     print(target.shape)


# for i,(data,target) in enumerate(val1):
#     print(data.shape)
#     print(target.shape)

file = h5py.File('../dataset/nskt16000_1024/train/nskt_train.h5', 'r')
crop = 256 
space_scale = 4 
time_scale = 4
window_x = (1024-crop)//2
window_x_end = window_x + crop
window_y = (1024-crop)//2
window_y_end = window_y + crop
train_poly_hr = file['fields'][()][:,-1,window_x:window_x_end,window_y:window_y_end]
train_poly_lr = train_poly_hr[::4,::space_scale,::space_scale]
file.close()
print(train_poly_hr.shape)

t_hr = np.linspace(0,1,train_poly_hr.shape[0])
t_lr = np.linspace(0,1,train_poly_lr.shape[0])
# up in time
cs = CubicSpline(t_lr, train_poly_lr)
train_poly_pred = torch.tensor(cs(t_hr)).unsqueeze(1)
pred = F.interpolate(train_poly_pred , size = (256,256), mode = 'bicubic', align_corners=False)
print(pred.shape)