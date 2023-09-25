from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import cmocean
import h5py
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from src.models import *
import torchaudio
from scipy.stats import pearsonr
from scipy.ndimage import zoom
import pyfftw

CONV_FILTER_y4 =  [[[[    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0],
           [1/12, -8/12,  0,  8/12, -1/12],
           [    0,   0,   0,   0,     0],
           [    0,   0,   0,   0,     0]]]]
CONV_FILTER_x4 = [[[[    0,   0,   1/12,   0,     0],
           [    0,   0,   -8/12,   0,     0],
           [    0,   0,   0,   0,     0],
           [    0,   0,   8/12,   0,     0],
           [    0,   0,   -1/12,   0,     0]]]]
CONV_FILTER_x2 = [[[[    0,   -1/2,   0],
                    [    0,   0,   0],
                    [     0,   1/2,   0]]]]

CONV_FILTER_y2 = [[[[    0,   0,   0],
                    [    -1/2,   0,   1/2],
                    [     0,   0,   0]]]]

DECAY_TURB_ = {"train_path":'../Decay_Turbulence_small/train/Decay_turb_small_128x128_1625.h5',
               "test_path":'../Decay_Turbulence_small/test/Decay_turb_small_128x128_79.h5',
               "dt":0.02,
               "resol":(128,128),
               }

RBC_= {"train_path":'../RBC_small/train/RBC_small_702_s2.h5',
               "test_path":'../RBC_small/test/RBC_small_33_s2.h5',
               "dt":0.01,
               "resol":(256,64)}

BURGER_2D_ = {"train_path":'../Burgers_2D_small/train/Burgers2D_128x128_1573.h5',
                "test_path":'../Burgers_2D_small/test/Burgers2D_128x128_79.h5',
                "dt":0.001,
                "resol":(128,128)}
              
DATA_INFO_TEST = {"decay_turb":['../Decay_Turbulence_small/test/Decay_turb_small_128x128_79.h5', 0.02],
                 "burger2d": ["../Burgers_2D_small/test/Burgers2D_128x128_79.h5",0.001],
                 "rbc": ["../RBC_small/test/RBC_small_33_s2.h5",0.01]}

class Conv2dDerivative(nn.Module):
    def __init__(self, DerFilter, resol, kernel_size=3, name=''):
        super(Conv2dDerivative, self).__init__()

        self.resol = resol  # constant in the finite difference
        self.name = name
        self.input_channels = 1
        self.output_channels = 1
        self.kernel_size = kernel_size

        self.padding = int((kernel_size - 1) // 2)
        self.filter = nn.Conv2d(self.input_channels, self.output_channels, self.kernel_size, 
            1, padding=0, bias=False)

        # Fixed gradient operator
        self.filter.weight = nn.Parameter(torch.FloatTensor(DerFilter), requires_grad=False)  

    def forward(self, input):
        derivative = self.filter(input)
        return derivative / self.resol    




class Fluid_Properties_Calculator(nn.Module):
    def __init__(self,data_name):
        self.data_name = data_name
        self.data_path = DATA_INFO_TEST[data_name][0]
    def _conv_operator(self,kernel_size,dxdy):
        if kernel_size ==5:
            self.dx = Conv2dDerivative(
                DerFilter = CONV_FILTER_x4,
                resol = dxdy[0],
                kernel_size = 5,
                name = 'dx_operator').cuda()

            self.dy = Conv2dDerivative(
                DerFilter = CONV_FILTER_y4,
                resol = dxdy[1],
                kernel_size = 5,
                name = 'dy_operator').cuda()  
        elif kernel_size ==3:
            self.dx = Conv2dDerivative(
                DerFilter = CONV_FILTER_x2,
                resol = dxdy[0],
                kernel_size = 3,
                name = 'dx_operator').cuda()

            self.dy = Conv2dDerivative(
                DerFilter = CONV_FILTER_y2,
                resol = dxdy[1],
                kernel_size = 3,
                name = 'dy_operator').cuda() 
    def _spectral_operator(nx,ny,u):
        
        '''
        compute the gradient of u using spectral differentiation
        
        Inputs
        ------
        nx,ny : number of grid points in x and y direction on fine grid
        u : 2D solution field 
        
        Output
        ------
        ux : du/dx
        uy : du/dy
        '''
        
        ux = np.empty((nx+1,ny+1))
        uy = np.empty((nx+1,ny+1))
        
        uf = np.fft.fft2(u[0:nx,0:ny])

        kx = np.fft.fftfreq(nx,1/nx)
        ky = np.fft.fftfreq(ny,1/ny)
        
        kx = kx.reshape(nx,1)
        ky = ky.reshape(1,ny)
        
        uxf = 1.0j*kx*uf
        uyf = 1.0j*ky*uf 
        
        ux[0:nx,0:ny] = np.real(np.fft.ifft2(uxf))
        uy[0:nx,0:ny] = np.real(np.fft.ifft2(uyf))
        
        # periodic bc
        ux[:,ny] = ux[:,0]
        ux[nx,:] = ux[0,:]
        ux[nx,ny] = ux[0,0]
        
        # periodic bc
        uy[:,ny] = uy[:,0]
        uy[nx,:] = uy[0,:]
        uy[nx,ny] = uy[0,0]
        
        return ux,uy     
    def get_divergence_conv(self,uv:torch.Tensor,kernel_size=3,dxdy=None,mean = True):
        """
        Compute the divergence of a 2D vector field.
        
        Args:
            uv (torch.Tensor): Input tensor of shape [B, 2, H, W].
            kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
            dxdy (tuple, optional): Tuple containing resolution in x and y directions. Default is based on uv dimensions.
            mean (bool, optional): If True, return the mean divergence, else return the tensor. Default is True.
        
        Returns:
            torch.Tensor or float: Divergence tensor of shape [H, W] or single value if mean=True.
        """
        if dxdy is None:
            dxdy = (1/uv.shape[-2],1/uv.shape[-1])
        self._conv_operator(kernel_size,dxdy)
        u = uv[:,0].unsqueeze(1)
        v = uv[:,1].unsqueeze(1)
        div = self.dx(u) + self.dy(v)
        if mean:
            div = div.mean().item()
        return div
    def get_vorticity_conv(self,uv:torch.Tensor,kernel_size=3,dxdy=None):
        """
        Compute the vorticity of a 2D vector field.
        
        Args:
            uv (torch.Tensor): Input tensor of shape [B, 2, H, W].
            kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
            dxdy (tuple, optional): Tuple containing resolution in x and y directions. Default is based on uv dimensions.
        
        Returns:
            torch.Tensor: Vorticity tensor.
        """
        if dxdy is None:
            dxdy = (1/uv.shape[-2],1/uv.shape[-1])
        self._conv_operator(kernel_size,dxdy)
        u = uv[:,0].unsqueeze(1)
        v = uv[:,1].unsqueeze(1)
        vorticity = self.dx(v) - self.dy(u)
        return vorticity
    def get_energy_spectrum(self,uv):
        import numpy as np
        import matplotlib.pyplot as plt
        import time
        from math import sqrt
        data = uv
        # print ("Computing spectrum... ",localtime)
        N = data.shape[-1]
        M= data.shape[-2]
        print("N =",N)
        print("M =",M)
        eps = 1e-32 # to void log(0)
        U = data[:,0].mean(axis=0)
        V = data[:,1].mean(axis=0)
        amplsU = abs(np.fft.fftn(U)/U.size)
        amplsV = abs(np.fft.fftn(V)/V.size)
        print(f"amplsU.shape = {amplsU.shape}")
        EK_U  = amplsU**2
        EK_V  = amplsV**2 
        EK_U = np.fft.fftshift(EK_U)
        EK_V = np.fft.fftshift(EK_V)
        sign_sizex = np.shape(EK_U)[0]
        sign_sizey = np.shape(EK_U)[1]
        box_sidex = sign_sizex
        box_sidey = sign_sizey
        box_radius = int(np.ceil((np.sqrt((box_sidex)**2+(box_sidey)**2))/2.)+1)
        centerx = int(box_sidex/2)
        centery = int(box_sidey/2)
        # print ("box sidex     =",box_sidex) 
        # print ("box sidey     =",box_sidey) 
        # print ("sphere radius =",box_radius )
        # print ("centerbox     =",centerx)
        # print ("centerboy     =",centery)
        EK_U_avsphr = np.zeros(box_radius,)+eps ## size of the radius
        EK_V_avsphr = np.zeros(box_radius,)+eps ## size of the radius
        for i in range(box_sidex):
            for j in range(box_sidey):          
                wn =  int(np.round(np.sqrt((i-centerx)**2+(j-centery)**2)))
                EK_U_avsphr[wn] = EK_U_avsphr [wn] + EK_U [i,j]
                EK_V_avsphr[wn] = EK_V_avsphr [wn] + EK_V [i,j]     
        EK_avsphr = 0.5*(EK_U_avsphr + EK_V_avsphr)
        realsize = len(np.fft.rfft(U[:,0]))
        TKEofmean_discrete = 0.5*(np.sum(U/U.size)**2+np.sum(V/V.size)**2)
        TKEofmean_sphere   = EK_avsphr[0]
        total_TKE_discrete = np.sum(0.5*(U**2+V**2))/(N*M) # average over whole domaon / divied by total pixel-value
        total_TKE_sphere   = np.sum(EK_avsphr)
        result_dict = {
        "Real Kmax": realsize,
        "Spherical Kmax": len(EK_avsphr),
        "KE of the mean velocity discrete": TKEofmean_discrete,
        "KE of the mean velocity sphere": TKEofmean_sphere,
        "Mean KE discrete": total_TKE_discrete,
        "Mean KE sphere": total_TKE_sphere
        }
        return realsize, EK_avsphr,result_dict
    
    def get_energy_spectrum_simple(self,uv,normalize):
        # adpat from Ray wang
        import radialProfile
        """Convert TKE field to spectrum"""
        tke = uv**2
        tke = tke.sum(axis=1)
        sp = np.fft.fft2(tke)
        sp = np.fft.fftshift(sp)
        sp = np.real(sp*np.conjugate(sp))
        sp1D = radialProfile.azimuthalAverage(sp)
            tensor = inverse_seqs(tensor)
    spec = np.array([tke2spectrum(TKE(tensor[i])) for i in range(tensor.shape[0])])
    return np.mean(spec, axis = 0), np.std(spec, axis = 0)

        return sp1D 

    def get_vorticity_energy_spectrum(nx,ny,w):
    
        '''
        Computation of energy spectrum and maximum wavenumber from vorticity field
        
        Inputs
        ------
        nx,ny : number of grid points in x and y direction
        w : vorticity field in physical spce (including periodic boundaries)
        
        Output
        ------
        en : energy spectrum computed from vorticity field
        n : maximum wavenumber
        '''
    
        epsilon = 1.0e-6

        kx = np.empty(nx)
        ky = np.empty(ny)
        
        kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
        kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

        ky[0:ny] = kx[0:ny]
        
        kx[0] = epsilon
        ky[0] = epsilon

        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        
        a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
        b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

        fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
        wf = fft_object(w[0:nx,0:ny]) 
        
        es =  np.empty((nx,ny))
        
        kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
        es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk
        
        n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1
        
        en = np.zeros(n+1)
        
        for k in range(1,n+1):
            en[k] = 0.0
            ic = 0
            ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
            ic = ii.size
            ii = ii+1
            jj = jj+1
            en[k] = np.sum(es[ii,jj])
                        
            en[k] = en[k]/ic
            
        return en, n