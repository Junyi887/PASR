
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.utils import weight_norm
from torch.autograd import Variable
import torch.nn.functional as F

torch.set_default_dtype(torch.float32)

def get_init_state(batch_size, hidden_channels, output_size, mode='coord'):
    '''initial hidden states for all convlstm layers'''
    # (b, c, h, w)

    num_layers = len(hidden_channels)
    initial_state = []
    if mode == 'coord':
        for i in range(num_layers):
            resolution = output_size[i][0]
            x, y = [np.linspace(-6654, 64, resolution+1)] * 2
            x, y = np.meshgrid(x[:-1], y[:-1])  # [32, 32]
            xy = np.concatenate((x[None, :], y[None, :]), 0) # [2, 32, 32]
            xy = np.repeat(xy, int(hidden_channels[i]/2), axis=0) # [c,h,w]
            xy = np.repeat(xy[None, :], batch_size[i], 0) # [b,c,h,w]
            xy = torch.tensor(xy, dtype=torch.float32)
            initial_state.append((xy.cuda(), xy.cuda()))

    elif mode == 'zero':
        for i in range(num_layers):
            (h0, c0) = (torch.zeros(batch_size[i], hidden_channels[i], output_size[i][0], 
                output_size[i][1]), torch.zeros(batch_size[i], hidden_channels[i], output_size[i][0], 
                output_size[i][1]))
            initial_state.append((h0,c0))

    elif mode == 'random':
        for i in range(num_layers):
            (h0, c0) = (torch.randn(batch_size[i], hidden_channels[i], output_size[i][0], 
                output_size[i][1]), torch.randn(batch_size[i], hidden_channels[i], output_size[i][0], 
                output_size[i][1]))
            initial_state.append((h0,c0))
    else:
        raise NotImplementedError

    return initial_state

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


def initialize_weights(module):

    c = 1
    if isinstance(module, nn.Conv2d):
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))

    if isinstance(module, nn.Conv1d):
        module.weight.data.uniform_(-c*np.sqrt(1 / (3 * 3 * 320)), 
            c*np.sqrt(1 / (3 * 3 * 320)))
     
    elif isinstance(module, nn.Linear):
        module.bias.data.zero_()


class ShiftMean(nn.Module):
    # note my data has shape [b,c,t,h,w]
    # data: [t,b,c,h,w]
    # channel: p, T, u, v
    def __init__(self, mean, std):
        super(ShiftMean, self).__init__()
        c = len(mean)
        self.mean = torch.Tensor(mean).view(1, c,1, 1, 1)
        self.std = torch.Tensor(std).view(1, c, 1, 1, 1)

    def forward(self, x, mode):
        if mode == 'sub':
            return (x - self.mean.cuda()) / self.std.cuda()
        elif mode == 'add':
            return x * self.std.cuda() + self.mean.cuda()
        else:
            raise NotImplementedError


class ConvLSTMCell(nn.Module):
    def __init__(self, input_feats, hidden_feats, input_kernel_size, input_stride, input_padding):
        super(ConvLSTMCell, self).__init__()

        self.hidden_feats = hidden_feats
        self.hidden_kernel_size = 3
        self.num_features = 3
        self.input_padding = input_padding
        self.padding = int((self.hidden_kernel_size - 1) / 2) # for the hidden state

        # input gate
        self.Wxi = nn.Conv2d(input_feats, hidden_feats, input_kernel_size, input_stride, 
            input_padding, bias=True, padding_mode='circular')
        self.Whi = nn.Conv2d(hidden_feats, hidden_feats, self.hidden_kernel_size, 
            1, padding=1, bias=False, padding_mode='circular')

        # forget gate
        self.Wxf = nn.Conv2d(input_feats, hidden_feats, input_kernel_size, input_stride, 
            input_padding, bias=True, padding_mode='circular')
        self.Whf = nn.Conv2d(hidden_feats, hidden_feats, self.hidden_kernel_size, 
            1, padding=1, bias=False, padding_mode='circular')

        # candidate gate
        self.Wxc = nn.Conv2d(input_feats, hidden_feats, input_kernel_size, input_stride, 
            input_padding, bias=True, padding_mode='circular')
        self.Whc = nn.Conv2d(hidden_feats, hidden_feats, self.hidden_kernel_size, 
            1, padding=1, bias=False, padding_mode='circular')

        # output gate
        self.Wxo = nn.Conv2d(input_feats, hidden_feats, input_kernel_size, input_stride, 
            input_padding, bias=True, padding_mode='circular')
        self.Who = nn.Conv2d(hidden_feats, hidden_feats, self.hidden_kernel_size, 
            1, padding=1, bias=False, padding_mode='circular')       

        # initialization
        nn.init.zeros_(self.Wxi.bias)
        nn.init.zeros_(self.Wxf.bias)
        nn.init.zeros_(self.Wxc.bias)
        self.Wxo.bias.data.fill_(1.0)

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h))
        ch = co * torch.tanh(cc)
        
        return ch, cc

    def init_hidden_tensor(self, prev_state):

        return (Variable(prev_state[0]).cuda(), Variable(prev_state[1]).cuda())


class ResBlock(nn.Module):
    def __init__(self, n_feats, expansion_ratio, res_scale=0.1):
        super(ResBlock, self).__init__()

        self.res_scale = res_scale
        self.conv1 = weight_norm(nn.Conv2d(n_feats, n_feats*expansion_ratio, kernel_size=3, 
            padding=1, padding_mode='circular'))
        self.conv2 = weight_norm(nn.Conv2d(n_feats*expansion_ratio, n_feats, kernel_size=3, 
            padding=1, padding_mode='circular'))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        s = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        x = s + self.res_scale * x

        return x

class temporal_sr(nn.Module):
    def __init__(self, t_upscale_factor):
        super(temporal_sr, self).__init__()

        self.t_upscale_factor = t_upscale_factor

    def forward(self, x):  

        t, b, c, h, w = x.shape  
        x = x.permute(1,3,4,2,0) # [b,h,w,c,t]
        x = x.contiguous().view(b*h*w, c, t)

        x = F.interpolate(x, size=self.t_upscale_factor+1, mode='linear', align_corners=True)   
        x = x.contiguous().view(b, h, w, c, 1+self.t_upscale_factor)
        x = x.permute(4,0,3,1,2) # [t,b,c,h,w]

        return x


class PhySR(nn.Module):
    def __init__(self, n_feats, n_layers, upscale_factor, shift_mean_paras, step=1, effective_step=[1],in_channels=3):

        super(PhySR, self).__init__()
        # n_layers: [n_convlstm, n_resblock]

        self.n_convlstm, self.n_resblock = n_layers
        self.t_up_factor, self.s_up_factor = upscale_factor
        self.mean, self.std = shift_mean_paras
        
        self.step = step
        self.effective_step = effective_step
        self._all_layers = []

        ################## temporal super-resolution ###################
        # temporal interpolation
        self.tsr = temporal_sr(self.t_up_factor)

        # temporal correction - convlstm
        for i in range(self.n_convlstm):
            name = 'convlstm{}'.format(i)
            cell = ConvLSTMCell(
                    input_feats=in_channels,
                    hidden_feats=n_feats,
                    input_kernel_size=3,
                    input_stride=1,
                    input_padding=1) 

            setattr(self, name, cell)
            self._all_layers.append(cell)

        ################## spatial super-resolution ###################
        body = [ResBlock(n_feats, expansion_ratio=4, res_scale=0.1) for _ in range(self.n_resblock)]
        tail = [weight_norm(nn.Conv2d(n_feats, in_channels*(self.s_up_factor ** 2), 
            kernel_size=3, padding=1, padding_mode='circular')), nn.PixelShuffle(self.s_up_factor)]  

        skip = [weight_norm(nn.Conv2d(in_channels, in_channels*(self.s_up_factor ** 2), kernel_size=5, stride=1,
            padding=2, padding_mode='circular')), nn.PixelShuffle(self.s_up_factor)]    

        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        # initialize weights
        self.apply(initialize_weights)

        # shiftmean
        self.shift_mean = ShiftMean(self.mean, self.std)    

    def forward(self, x, initial_state):
        # input: [t,b,c,h,w] 
        x = self.shift_mean(x, mode='sub')
        x = x.permute(2,0,1,3,4) # [b,c,t,h,w] --> [t,b,c,h,w]
        tt,bb,cc,hh,ww = x.shape
        internal_state = []
        outputs = []
        # normalize
        # temporal super-resolution
        x = self.tsr(x) 
        for step in range(self.step):
            # input:[t,b,c,h,w]
            xt = x[step,...]
            # skip connection
            s = self.skip(xt)
            # temporal correction
            for i in range(self.n_convlstm):
                name = 'convlstm{}'.format(i)
                if step == 0:
                    (h, c) = getattr(self, name).init_hidden_tensor(
                        prev_state = (torch.randn(bb, 32, x.shape[-2], 
                x.shape[-1]), torch.randn(bb, 32, x.shape[-2], 
                x.shape[-1])))
                    internal_state.append((h,c))
                    
                # one-step forward
                (h, c) = internal_state[i]
                xt, new_c = getattr(self, name)(xt, h, c)
                internal_state[i] = (xt, new_c)  

            # spatial super-resolution
            xt = self.body(xt)
            xt = self.tail(xt)
            # residual connection
            xt += s # [b,c,h,w]
            xt = xt.view(bb, cc, hh*4, ww*4) 
            
            if step in self.effective_step:
                outputs.append(xt)    
        # outputs = torch.cat(tuple(outputs), dim=1)
        out = torch.stack(outputs, dim=2)
        out = self.shift_mean(out, mode='add')
        return out

