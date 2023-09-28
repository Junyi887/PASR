import torch
import torch.nn as nn
from .basics import SpectralConv3d
import torch.nn.functional as F
class ShiftMean(nn.Module):
    # note my data has shape [b,c,t,h,w]
    # data: [t,b,c,h,w]
    # channel: p, T, u, v
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        c = len(mean)
        self.mean = torch.Tensor(mean).view(1, c, 1, 1, 1) # B, C, T, X, Y
        self.std = torch.Tensor(std).view(1, c, 1, 1, 1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean.cuda()) / self.std.cuda()
        elif mode == 'add':
            return x * self.std.cuda() + self.mean.cuda()
        else:
            raise NotImplementedError
        
class FNO3D(nn.Module):
    def __init__(self, 
                 modes1, modes2, modes3,
                 HR_shape,
                 width=16, 
                 fc_dim=128,
                 layers=None,
                 in_dim=3, out_dim=3,
                 act='gelu', 
                 pad_ratio=[0., 0.],mean=[0],std =[1]):
        '''
        Args:
            modes1: list of int, first dimension maximal modes for each layer
            modes2: list of int, second dimension maximal modes for each layer
            modes3: list of int, third dimension maximal modes for each layer
            layers: list of int, channels for each layer
            fc_dim: dimension of fully connected layers
            in_dim: int, input dimension
            out_dim: int, output dimension
            act: {tanh, gelu, relu, leaky_relu}, activation function
            pad_ratio: the ratio of the extended domain
        '''
        super(FNO3D, self).__init__()

        if isinstance(pad_ratio, float):
            pad_ratio = [pad_ratio, pad_ratio]
        else:
            assert len(pad_ratio) == 2, 'Cannot add padding in more than 2 directions.'

        self.pad_ratio = pad_ratio
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.pad_ratio = pad_ratio
        self.shift_mean = ShiftMean(mean, std)  
        if layers is None:
            self.layers = [width] * 4
        else:
            self.layers = layers
        self.fc0 = nn.Linear(in_dim, self.layers[0])

        self.sp_convs = nn.ModuleList([SpectralConv3d(
            in_size, out_size, mode1_num, mode2_num, mode3_num)
            for in_size, out_size, mode1_num, mode2_num, mode3_num
            in zip(self.layers, self.layers[1:], self.modes1, self.modes2, self.modes3)])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(self.layers, self.layers[1:])])

        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)
        self.hr_shape = HR_shape
    def forward(self, x):
        '''
        Args:
            x_LR: (batchsize, C, T, X, Y)

        Returns:
            x_HR pred: (batchsize, x_grid, y_grid, t_grid, 3)

        '''
        # dimension in LR
        B,C,T,H,W = x.shape
        x = F.interpolate(x, size=(self.hr_shape[2],self.hr_shape[3],self.hr_shape[4]), mode='trilinear', align_corners=False)
        x.shift_mean(x, 'sub')
        x = x.permute(0, 2, 3, 4, 1) # from bcxyz to bxyzc
        size_z = x.shape[-2]
        length = len(self.ws)
        batchsize = x.shape[0]
        x = self.fc0(x)
        x = x.permute(0, 4, 1, 2, 3)
        size_x, size_y, size_z = x.shape[-3], x.shape[-2], x.shape[-1]

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x.view(batchsize, self.layers[i], -1)).view(batchsize, self.layers[i+1], size_x, size_y, size_z)
            x = x1 + x2
            if i != length - 1:
                x = self.act(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.permute(0, 4, 1, 2, 3)
        x.shift_mean(x, 'add')
        return x
    



def _get_act(act):
    if act == 'tanh':
        func = F.tanh
    elif act == 'gelu':
        func = F.gelu
    elif act == 'relu':
        func = F.relu_
    elif act == 'elu':
        func = F.elu_
    elif act == 'leaky_relu':
        func = F.leaky_relu_
    else:
        raise ValueError(f'{act} is not supported')
    return func