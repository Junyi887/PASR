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
class ConvODEFunc(nn.Module):
    """Convolutional block modeling the derivative of ODE system.

    Parameters
    ----------
    device : torch.device

    img_size : tuple of ints
        Tuple of (channels, height, width).

    num_filters : int
        Number of convolutional filters.

    augment_dim: int
        Number of augmentation channels to add. If 0 does not augment ODE.

    time_dependent : bool
        If True adds time as input, making ODE time dependent.

    non_linearity : string
        One of 'relu' and 'softplus'
    """
    def __init__(self, in_dim, num_filters, aug_dim_t=0,num_ode_layers = 4,final_tanh= False):
        super(ConvODEFunc, self).__init__()
        self.aug_dim_t = aug_dim_t
        self.channels = in_dim
        self.channels += aug_dim_t
        self.num_filters = num_filters
        self.num_layers = num_ode_layers

        if aug_dim_t > 0:
            self.conv_0 = Conv2dTime(self.channels, self.num_filters,
                                    kernel_size=1, stride=1, padding=0)
            self.conv_mid1 = Conv2dTime(self.num_filters, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv_mid2 = Conv2dTime(self.num_filters, self.num_filters,
                                    kernel_size=3, stride=1, padding=1)
            self.conv_last = Conv2dTime(self.num_filters, self.channels,
                                    kernel_size=1, stride=1, padding=0)
        else:
            self.conv_0 = nn.Conv2d(self.channels, self.num_filters,
                                   kernel_size=1, stride=1, padding=0)
            self.conv_mid1 = nn.Conv2d(self.num_filters, self.num_filters,
                                   kernel_size=3, stride=1, padding=1)
            self.conv_mid2 = nn.Conv2d(self.num_filters, self.num_filters,
                                   kernel_size=3, stride=1, padding=1)
            self.conv_last = nn.Conv2d(self.num_filters, self.channels,
                                   kernel_size=1, stride=1, padding=0)
        self.non_linearity = nn.ReLU(inplace=True)

    def forward(self, t, x):
        """
        Parameters
        ----------
        t : torch.Tensor
            Current time.

        x : torch.Tensor
            Shape (batch_size, input_dim)
        """
        if self.aug_dim_t >0:
            out = self.conv_0(t, x)
            out = self.non_linearity(out)
            out = self.conv_mid1(t, out)
            out = self.non_linearity(out)
            if self.num_layers == 4:
                out = self.conv_mid2(t,out)
                out = self.non_linearity(out)
            out = self.conv_last(t, out)
        else:
            out = self.conv_0(x)
            out = self.non_linearity(out)
            out = self.conv_mid1(out)
            if self.num_layers == 4:
                out = self.conv_mid2(out)
                out = self.non_linearity(out)
            out = self.non_linearity(out)
            out = self.conv_last(out)
        return out


        
class ODEBlock(nn.Module):
    """Solves ODE defined by odefunc.

    Parameters
    ----------
    device : torch.device

    odefunc : ODEFunc instance or anode.conv_models.ConvODEFunc instance
        Function defining dynamics of system.

    tol : float
        Error tolerance.
    """
    def __init__(self, odefunc, tol=1e-3,ode_method = 'euler'):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.tol = tol
        self.ode_method = ode_method

    def forward(self, x, eval_times=None):
        """Solves ODE starting from x.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        eval_times : None or torch.Tensor
            If None, returns solution of ODE at final time t=1. If torch.Tensor
            then returns full ODE trajectory evaluated at points in eval_times.
        """

        if eval_times is None:
            integration_time = torch.tensor([0, 1]).float().type_as(x)
        else:
            integration_time = eval_times.type_as(x)# initalize with a tensor
        if self.odefunc.aug_dim_t > 0:
            # Add augmentation
            batch_size, channels, height, width = x.shape
            aug = torch.zeros(batch_size, self.odefunc.aug_dim_t,
                                height, width).cuda()
            # Shape (batch_size, channels + augment_dim, height, width)
            x_aug = torch.cat([x, aug], 1)
        else:
            x_aug = x

        out = odeint(self.odefunc, x_aug, integration_time,
                         rtol=self.tol, atol=self.tol, method=self.ode_method)

        if eval_times is None:
            return out[1]  # Return only final time
        else:
            return out

    def trajectory(self, x, timesteps):
        """Returns ODE trajectory.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, self.odefunc.data_dim)

        timesteps : int
            Number of timesteps in trajectory.
        """
        integration_time = torch.linspace(0., 1., timesteps)
        return self.forward(x, eval_times=integration_time)

        
class Conv2dTime(nn.Conv2d):
    """
    Implements time dependent 2d convolutions, by appending the time variable as
    an extra channel.
    """
    def __init__(self, in_channels, *args, **kwargs):
        super(Conv2dTime, self).__init__(in_channels + 1, *args, **kwargs)

    def forward(self, t, x):
        # Shape (batch_size, 1, height, width)
        t_img = torch.ones_like(x[:, :1, :, :]) * t
        # Shape (batch_size, channels + 1, height, width)
        t_and_x = torch.cat([t_img, x], 1)
        return super(Conv2dTime, self).forward(t_and_x)
    
    
class PASR_ODE(nn.Module):
    """ PASR with ODE wrapper

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, upsampler='', resi_connection='1conv'
                 ,mean = [0],std = [1],  
                 ode_method = "euler",num_ode_layers = 4,time_update = "NODE",ode_kernel_size = 3,ode_padding = 1,aug_dim_t = None, **kwargs):
        super(PASR_ODE, self).__init__()
        
        self.time_update = time_update
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        # if in_chans == 3:
        #     self.mean = torch.Tensor(mean).view(1, 3, 1, 1)
        #     self.std = torch.Tensor(std).view(1, 3, 1, 1)
        # else:
        #     self.mean = torch.Tensor(mean).view(1, 1, 1, 1)
        #     self.std = torch.Tensor(std).view(1, 1, 1, 1)
        # shiftmean
        self.mean = mean
        self.std = std
        self.shiftMean_func = ShiftMean(self.mean, self.std)

        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual Swin Transformer blocks (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection

                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        #####################################################################################################
        ################################### 3, Neural ODE time interpolation ################################
        if aug_dim_t is not None: 
            self.ode_func = ConvODEFunc(embed_dim,embed_dim*2,aug_dim_t=aug_dim_t)
            self.ODEBlock = ODEBlock(self.ode_func,ode_method=ode_method)
        self.ode_method = ode_method
        self.aug_dim_t = aug_dim_t
        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim+aug_dim_t, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim+aug_dim_t, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim+aug_dim_t, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale %4 == 0:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif self.upsampler == 'shallowdecoder':
            if self.upscale == 8:
                self.shallowdecoder = nn.Sequential(nn.ConvTranspose2d(num_feat, num_feat, 4, 2, 1),
                                                    nn.LeakyReLU(inplace=True))
        else:
            # for image denoising and JPEG compression artifact reduction
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x,n_snapshots = 1,task_dt =1.0 ):
        # ode steps can be arbitary just enough to make it converge. 
        # n_snapshots should relates to task_dt.
        # during train task_dt is 1, then n_snapshots should be 1
        # if we want to intermediate snapshot change task dt to 0.5, then n_snapshots should be 2
        B,C,H,W = x.shape
        prediction_list = []
        x = self.check_image_size(x)
        x = self.shiftMean_func(x,"sub")
        x = self.conv_first(x)     #Shallow Feature Extraction
        z0 = self.conv_after_body(self.forward_features(x)) + x              #Deep Feature Extraction + x
        t = torch.linspace(0.0,task_dt,n_snapshots+1)
        if self.upsampler == 'pixelshuffle':
            # load initial condition
            y_t = self.ODEBlock(z0,eval_times = t)
            T,B,C_l,H_l,W_l= y_t.shape
            y_t= y_t.permute(1,0,2,3,4) # to B,T,C,H,W
            for i in range(T):
                y = self.conv_before_upsample(y_t[:,i])                 #HQ Image Reconstruction
                y = self.conv_last(self.upsample(y))  
                y = self.shiftMean_func(y,"add")
                prediction_list.append(y)
            prediction = torch.stack(prediction_list,dim = 1) # B,T,C,H,W
            return prediction
        
        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
            
        # elif self.upsampler == 'shallowdecoder':
        #     # for real-world SR
        #     x = self.conv_first(x)
        #     x = self.conv_after_body(self.forward_features(x)) + x
        #     x = self.conv_before_upsample(x)
        #     x = self.shallowdecoder(x)
        #     if self.upscale == 4:
        #         x = self.shallowdecoder(x)
        #     if self.upscale == 8:
        #         x = self.shallowdecoder(x)
        #         x = self.shallowdecoder(x)
        #     x = self.conv_last(self.lrelu(self.conv_hr(x)))
        # else:
        #     # for image denoising and JPEG compression artifact reduction
        #     x_first = self.conv_first(x)
        #     res = self.conv_after_body(self.forward_features(x_first)) + x_first
        #     x = x + self.conv_last(res)

        

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops



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



