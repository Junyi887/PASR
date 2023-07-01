import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import h5py
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

def getData(data_name = "nskt_16k", data_path = "../superbench/datasets/nskt16000_1024" , upscale_factor = 4,timescale_factor = 4, noise_ratio = 0.0, crop_size = 1024, method = "bicubic", batch_size = 1, std = [0.6703, 0.6344, 8.3615]):  
    '''
    Loading data from four dataset folders: (a) nskt_16k; (b) nskt_32k; (c) cosmo; (d) era5.
    Each dataset contains: 
        - 1 train dataset, 
        - 2 validation sets (interpolation and extrapolation), 
        - 2 test sets (interpolation and extrapolation),
        
    ===
    std: the channel-wise standard deviation of each dataset, list: [#channels]
    '''
    num_snapshots = 2

    train_loader = get_data_loader(data_name, data_path, '/train', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std)
    val1_loader = get_data_loader(data_name, data_path, '/train', "val", upscale_factor, timescale_factor//2,num_snapshots,noise_ratio, crop_size, method, batch_size, std)
    val2_loader = get_data_loader(data_name, data_path, '/valid_1', "val", upscale_factor,timescale_factor//2,num_snapshots,noise_ratio, crop_size, method, batch_size, std) 
    test1_loader = get_data_loader(data_name, data_path, '/valid_2', "test", upscale_factor,timescale_factor//4, num_snapshots*2,noise_ratio, crop_size, method, batch_size, std)
    test2_loader = get_data_loader(data_name, data_path, '/valid_1', "test", upscale_factor,timescale_factor//4, num_snapshots*2, noise_ratio, crop_size, method, batch_size, std)
    
    return train_loader, val1_loader, val2_loader, test1_loader, test2_loader 

def get_data_loader(data_name, data_path, data_tag, state, upscale_factor, timescale_factor, num_snapshots,noise_ratio, crop_size, method, batch_size, std):
    
    transform = torch.from_numpy

    if data_name in ['nskt_16k']:
        dataset = GetFluidDataset(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method) 

    if state == "train":
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(dataset,
                            batch_size = int(batch_size),
                            num_workers = 0, # TODO: make a param
                            shuffle = shuffle, 
                            sampler = None,
                            drop_last = True,
                            pin_memory = torch.cuda.is_available())

    return dataloader



class GetFluidDataset(Dataset):
    '''Dataloader class for NSKT and cosmo datasets'''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method):
        self.location = location
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self._get_files_stats()
        self.crop_size = crop_size
        self.crop_transform = transforms.CenterCrop(crop_size)
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        if method == "bicubic":
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                                             transforms.Resize((int(self.crop_size/upscale_factor),int(self.crop_size/upscale_factor)),Image.BICUBIC) ]) # subsampling the image (half size)
        elif method == "gaussian_blur":
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
                                                       transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))])
        elif method == "uniform":
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
                                        ])
        self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size) # since it's the target, we keep its original quality
                                        ])
    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['fields'].shape[0]
            self.n_in_channels = _f['fields'].shape[1]
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_files * self.n_samples_per_file
        self.files = [None for _ in range(self.n_files)]
        print("Number of samples per file: {}".format(self.n_samples_per_file))
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['fields']  

    def __len__(self):
        return self.n_samples_total-self.timescale_factor*self.num_snapshots

    def __getitem__(self, global_idx):
        # print("batch_idx: {}".format(batch_idx))
        y_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        y = self.transform(self.files[file_idx][local_idx]) # from numpy to torch
        y = self.target_transform(y) # cropping the image
        y = y[-1].unsqueeze(0) # get vorticity and adding the channel dimension
        X = self.get_X(y) # getting the input
        # getting the future samples
        y_list.append(y)
        for i in range(1, self.num_snapshots+1):
            file_idx, local_idx_future = self.get_indices(global_idx + i*self.timescale_factor)
            #open image file for future sample
            if self.files[file_idx] is None:
                self._open_file(file_idx)
            y = self.transform(self.files[file_idx][local_idx_future])
            y = self.target_transform(y)
            y = y[-1].unsqueeze(0) # adding the channel dimension and get vorticity
            y_list.append(y)
        y = torch.stack(y_list,dim = 0) 
        return X,y
        # if self.state == "train":
        #     return X,y
        #     #return X,torch.stack([y[0,...],y[self.timescale_factor/2,...],y[-1]],dim= 0)
        # if self.state == "val":
        #     return X,y
        #     #return X,torch.stack([y[0,...],y[-1]],dim= 0)
        # if self.state == 'test':
        #     return X,y
        # return 0,0
           # raise IndexError("Index out of bound or not valid due to timescale factor")



    def get_indices(self, global_idx):
        file_idx = int(global_idx/self.n_samples_per_file)  # which file we are on
        local_idx = int(global_idx % self.n_samples_per_file)  # which sample in that file we are on 

        return file_idx, local_idx

    def get_X(self, y):
        if self.method == "uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
        elif self.method == "noisy_uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
            X = X + self.noise_ratio * self.std * torch.randn(X.shape)
        elif self.method == "bicubic":
            X = self.input_transform(y)
        else:
            raise ValueError(f"Invalid method: {self.method}")
        #TODO: add gaussian blur
        return X







if __name__ == "__main__":
    train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(batch_size= 1)
    for idx, (input,target) in enumerate (train_loader):
        input = input
        target = target
    print(input.shape)
    print(target.shape)
    print(idx)
    for idx, (input,target) in enumerate (val1_loader):
        input = input
        target = target
    print(input.shape)
    print(target.shape)
    for idx, (input,target) in enumerate (test1_loader):
        input = input
        target = target
    print(input.shape)
    print(target.shape)
    list = []
    for i in range(2):
        x = torch.rand(1,1,128)
        list.append(x)
    x = torch.stack(list,dim = 0)
    print(x.shape)
