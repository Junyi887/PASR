# Feature extraction
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import savemat
from scipy.io import loadmat
import torch.nn.init as init
from math import log10
import torch.optim as optim
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
from torchdiffeq import odeint
# from SwinIR_basics import Mlp
# from SwinIR_basics import window_partition
# from SwinIR_basics import WindowAttention
# from SwinIR_basics import SwinTransformerBlock
from .SwinIR_basics import PatchUnEmbed
from .SwinIR_basics import PatchEmbed
# from SwinIR_basics import PatchMerging
# from SwinIR_basics import BasicLayer
from .SwinIR_basics import RSTB
from .SwinIR_basics import Upsample
from .SwinIR_basics import UpsampleOneStep



class Conv_ODE(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters, aug_dim_t=0, num_ode_layers = 4,final_tanh= False):
        super(Conv_ODE, self).__init__()
    def forward(self, x):
        pass



class encoder_layer(nn.Module):
    def __init__(self, in_dim, out_dim,kernel_size = 3 , stride = 2):
        super(encoder_layer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride = stride, padding = 1, padding_mode= 'circular',bias = True),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)

class decoder_layer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(decoder_layer, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, kernel_size = 4, stride = 2,padding = 1, bias = True),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.layer(x)



class ShiftMean(nn.Module):
    # data: [t,c,h,w]
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        c = len(mean)
        self.mean = torch.Tensor(mean).view(1,c,1,1)
        self.std = torch.Tensor(std).view(1,c,1,1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        elif mode == 'add':
            return x * self.std.to(x.device) + self.mean.to(x.device)
        else:
            raise NotImplementedError



