import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import h5py
import torchvision.transforms as transforms
import torch.nn.functional
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
#"../superbench/datasets/nskt16000_1024"
#../datasets/rbc_diff_IC/rbc_IC1
def getData(data_name = "rbc_diff_IC", data_path =  "../rbc_diff_IC/rbc_10IC",
             upscale_factor= 4,timescale_factor = 1, num_snapshots = 20,
             noise_ratio = 0.0, crop_size = 128, method = "bicubic", 
             batch_size = 1, std = [0.6703, 0.6344, 8.3615],in_channels = 1):  
    
    # data_name, data_path, data_tag, state, upscale_factor, timescale_factor, num_snapshots,noise_ratio, crop_size, method, batch_size, std
        #To do swap and change 
    if data_name == "climate":
        dataset = GetClimateDatasets(data_path, "train",torch.from_numpy , upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
        print("Climate Loader")
        train_set,val_set,test_set = random_split(dataset,[0.8,0.1,0.1],generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        val1_loader= DataLoader(val_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        val2_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        test1_loader = val1_loader
        test2_loader = val2_loader
        return train_loader,val1_loader,val2_loader,test1_loader,test2_loader
    elif data_name == "climate_sequence":
        dataset = GetClimateDatasets_special(data_path, "train",torch.from_numpy , upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
        print("Climate Loader")
        train_set,val_set,test_set = random_split(dataset,[0.8,0.1,0.1],generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        val1_loader= DataLoader(val_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        val2_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        test1_loader = val1_loader
        test2_loader = val2_loader
        return train_loader,val1_loader,val2_loader,test1_loader,test2_loader
    else:
        train_loader = get_data_loader(data_name, data_path, '/train', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels)
        val1_loader = get_data_loader(data_name, data_path, '/val', "test", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels)
        val2_loader = get_data_loader(data_name, data_path, '/test', "no_roll_out", upscale_factor,timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std, in_channels)
        test1_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor,num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels)
        test2_loader = get_data_loader(data_name, data_path, '/test', "no_roll_out", upscale_factor,timescale_factor, num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels)
   
        return train_loader,val1_loader,val2_loader,test1_loader,test2_loader
    
def get_data_loader(data_name, data_path, data_tag, state, upscale_factor, timescale_factor, num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels=1):
    
    transform = torch.from_numpy
    print("Data Name: ", data_name)
    if ("FNO" in data_name) or ("ConvLSTM" in data_name): 
        print("Sequence Loader for FNO and ConvLSTM")
        dataset = Special_Loader_Fluid(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels) 
    elif "lrsim" in data_name:
        print("Loader for NODE with low res as input")
        dataset = GetDataset_diffIC_LowRes(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
    elif "coord" in data_name:
        print("Loader for FNO_v2 with coordinates cat as input")
        dataset = FNO_Special_Loader_Fluid(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
    elif "sequenceLR" in data_name:
        print("Normal Loader for ConvLSTM Low Res")
        dataset = GetDataset_diffIC_LowRes_sequence(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
    else:
        print("Normal Loader for NODE ")
        dataset = GetDataset_diffIC_NOCrop(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels) 
    
    if state == "train":
        shuffle = True
        drop_last = True
    else:
        shuffle = False
        drop_last = False

    dataloader = DataLoader(dataset,
                            batch_size = int(batch_size),
                            num_workers = 2, # TODO: make a param
                            shuffle = shuffle, 
                            sampler = None,
                            drop_last = drop_last,
                            pin_memory = False)
    return dataloader


class GetDataset_diffIC_NOCrop(Dataset):
    '''Dataloader from different initial conditions
    It loads single low-resolution image as a input and gives following high-resolution images as targets
    '''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method,in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        if method == "bicubic":
            self.input_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False) # TODO: compatibility issue for antialias='warn' check torch version
        elif method == "gaussian_blur":
            self.input_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))
        
    def _get_files_stats(self):
        # larger dt = 0.1
        self.files_paths = glob.glob(self.location + "/*.h5") #only take s9
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['tasks']["u"].shape[0]
            self.img_shape_x = _f['tasks']["u"].shape[1]
            self.img_shape_y = _f['tasks']["u"].shape[2]

        final_index = (self.n_samples_per_file-1)//self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
            print("no_roll_out",self.idx_matrix)
        else:
            self.idx_matrix = self.generate_toeplitz(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        if self.num_snapshots != self.idx_matrix.shape[1] -1:
            raise ValueError(f"Invalid number of snapshots: {self.num_snapshots} vs {self.idx_matrix.shape[1]}")
        self.n_samples_total = self.n_files*self.input_per_file
        print(self.n_samples_total)
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        # each file must have same number of files, otherwise it will be wrong
        print("Found data at path {}. Number of examples total: {}. To-use data per trajectory: {}  Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_per_file, self.input_per_file,self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['tasks']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        y_list = []
        t_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # lr 
        if local_idx !=self.idx_matrix[global_idx%self.input_per_file][0]:
                raise ValueError(f"Invalid Input index: {local_idx} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        if self.n_in_channels ==3:
            w,u,v = self.files[file_idx]["vorticity"][local_idx],self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            w,u,v = self.transform(w),self.transform(u),self.transform(v)
            y = torch.stack((w,u,v),dim = 0)
        elif self.n_in_channels ==2:
            u,v = self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            u,v = self.transform(u),self.transform(v)
            y = torch.stack((u,v),dim = 0)
        elif self.n_in_channels ==1:
            w = self.files[file_idx]["vorticity"][local_idx]
            w = self.transform(w)
            y = w.unsqueeze(0)
        X = self.get_X(y)
        y_list.append(y)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i*self.timescale_factor
            if local_idx_future !=self.idx_matrix[global_idx%self.input_per_file][i]:
                raise ValueError(f"Invalid target index: {local_idx_future} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
            if self.n_in_channels ==3:
                w,u,v = self.files[file_idx]["vorticity"][local_idx_future],self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                w,u,v = self.transform(w),self.transform(u),self.transform(v)
                y = torch.stack((w,u,v),dim = 0)
            elif self.n_in_channels ==2:
                u,v = self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                u,v = self.transform(u),self.transform(v)
                y = torch.stack((u,v),dim = 0)
            elif self.n_in_channels ==1:
                w = self.files[file_idx]["vorticity"][local_idx_future]
                w = self.transform(w)
                y = w.unsqueeze(0)
            y_list.append(y) 
            # t_list.append(t)
        y = torch.stack(y_list,dim = 0) 

        # t = torch.stack(t_list,dim = 0) 
        return X,y

    def get_indices(self, global_idx):
        if self.state =="no_roll_out":
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor *self.num_snapshots # which sample in
        else:
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor  # which sample in that file we are on 
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
    
    @staticmethod
    def generate_toeplitz(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
        rows = final_index - cols + 2
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix such that it becomes a Toeplitz matrix
        for i in range(rows):
            for j in range(cols):
                value = i + j
                matrix[i, j] = min(value, final_index)
                    
        return matrix
    @staticmethod
    def generate_test_matrix(cols:int, final_index:int):
        # Calculate the number of rows based on the final index and number of columns
        rows = (final_index + 1) // (cols - 1)
        
        # Check if an additional row is needed to reach the final index
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix according to the specified pattern
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                if current_value <= final_index:
                    matrix[i, j] = current_value
                    current_value += 1
            current_value -= 1  # Repeat the last element in the next row
                    
        return matrix[:-1,:]


class GetDataset_diffIC_LowRes(Dataset):
    '''Dataloader from different initial conditions
    It loads single low-resolution image as a input and gives following high-resolution images as targets
    '''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method,in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        
    def _get_files_stats(self):
        # larger dt = 0.1
        self.files_paths = glob.glob(self.location + "/*.h5") #only take s9
        # /Burger2D_*
        # /rbc_*_256/
        # Decay_turb
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['tasks']["u"].shape[0]
            self.img_shape_x = _f['tasks']["u"].shape[1]
            self.img_shape_y = _f['tasks']["u"].shape[2]
            self.img_shape_x_lr = _f['tasks']["u_lr"].shape[1]
            self.img_shape_y_lr = _f['tasks']["u_lr"].shape[2]
        final_index = (self.n_samples_per_file-1)//self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
            print(self.idx_matrix)
        else:
            self.idx_matrix = self.generate_toeplitz(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        if self.num_snapshots != self.idx_matrix.shape[1] -1:
            raise ValueError(f"Invalid number of snapshots: {self.num_snapshots} vs {self.idx_matrix.shape[1]}")
        self.n_samples_total = self.n_files*self.input_per_file
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        
        # each file must have same number of files, otherwise it will be wrong
        print(f"find files at location {self.location}. Number of examples total: {self.n_samples_per_file}. To-use data per trajectory: {self.input_per_file}  Image Shape: {self.img_shape_x} x {self.img_shape_y} x {self.n_in_channels} LR Image Shape: {self.img_shape_x_lr} x {self.img_shape_y_lr} x {self.n_in_channels}")

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['tasks']
        # self.times[file_idx] = _file['scales/sim_time'] disabeld and add to the model with aug convolution

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        y_list = []
        t_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # lr 
        if local_idx !=self.idx_matrix[global_idx%self.input_per_file][0]:
                raise ValueError(f"Invalid Input index: {local_idx} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        if self.n_in_channels ==3:
            w,u,v = self.files[file_idx]["vorticity"][local_idx],self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            w_lr,u_lr,v_lr = self.files[file_idx]["vorticity_lr"][local_idx],self.files[file_idx]["u_lr"][local_idx],self.files[file_idx]["v_lr"][local_idx]
            w,u,v = self.transform(w),self.transform(u),self.transform(v)
            w_lr,u_lr,v_lr = self.transform(w_lr),self.transform(u_lr),self.transform(v_lr)
            y = torch.stack((w,u,v),dim = 0)
            X = torch.stack((w_lr,u_lr,v_lr),dim = 0)
        elif self.n_in_channels ==2:
            u,v = self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            u_lr,v_lr = self.files[file_idx]["u_lr"][local_idx],self.files[file_idx]["v_lr"][local_idx]
            u,v = self.transform(u),self.transform(v)
            u_lr,v_lr = self.transform(u_lr),self.transform(v_lr)
            y = torch.stack((u,v),dim = 0)
            X = torch.stack((u_lr,v_lr),dim = 0)
        elif self.n_in_channels ==1:
            w = self.files[file_idx]["vorticity"][local_idx]
            w_lr = self.files[file_idx]["vorticity_lr"][local_idx]
            w = self.transform(w)
            w_lr = self.transform(w_lr)
            y = w.unsqueeze(0)
            X = w_lr.unsqueeze(0)
        y_list.append(y)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i*self.timescale_factor
            if local_idx_future !=self.idx_matrix[global_idx%self.input_per_file][i]:
                raise ValueError(f"Invalid target index: {local_idx_future} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
            
            if self.n_in_channels ==3:
                w,u,v = self.files[file_idx]["vorticity"][local_idx_future],self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                w,u,v = self.transform(w),self.transform(u),self.transform(v)
                y = torch.stack((w,u,v),dim = 0)
            elif self.n_in_channels ==2:
                u,v = self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                u,v = self.transform(u),self.transform(v)
                y = torch.stack((u,v),dim = 0)
            elif self.n_in_channels ==1:
                w = self.files[file_idx]["vorticity"][local_idx_future]
                w = self.transform(w)
                y = w.unsqueeze(0)
            y_list.append(y) 
            # t_list.append(t)
        y = torch.stack(y_list,dim = 0) 
        return X,y

    def get_indices(self, global_idx):
        if self.state =="no_roll_out":
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor*self.num_snapshots  # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor  # which sample in that file we are on 
        return file_idx, local_idx
    
    @staticmethod
    def generate_toeplitz(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
        rows = final_index - cols + 2
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix such that it becomes a Toeplitz matrix
        for i in range(rows):
            for j in range(cols):
                value = i + j
                matrix[i, j] = min(value, final_index)
                    
        return matrix
    @staticmethod
    def generate_test_matrix(cols:int, final_index:int):
        # Calculate the number of rows based on the final index and number of columns
        rows = (final_index + 1) // (cols - 1)
        
        # Check if an additional row is needed to reach the final index
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix according to the specified pattern
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                if current_value <= final_index:
                    matrix[i, j] = current_value
                    current_value += 1
            current_value -= 1  # Repeat the last element in the next row
                    
        return matrix[:-1,:]


class GetDataset_diffIC_LowRes_crop(Dataset):
    '''Dataloader from different initial conditions
    It loads single low-resolution image as a input and gives following high-resolution images as targets
    '''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method,in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        
    def _get_files_stats(self):
        # larger dt = 0.1
        self.files_paths = glob.glob(self.location + "/*.h5") #only take s9
        # /Burger2D_*
        # /rbc_*_256/
        # Decay_turb
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['tasks']["u"].shape[0]
            self.img_shape_x = _f['tasks']["u"].shape[1]
            self.img_shape_y = _f['tasks']["u"].shape[2]
            self.img_shape_x_lr = _f['tasks']["u_lr"].shape[1]
            self.img_shape_y_lr = _f['tasks']["u_lr"].shape[2]
        final_index = (self.n_samples_per_file-1)//self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
            print(self.idx_matrix)
        else:
            self.idx_matrix = self.generate_toeplitz(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        if self.num_snapshots != self.idx_matrix.shape[1] -1:
            raise ValueError(f"Invalid number of snapshots: {self.num_snapshots} vs {self.idx_matrix.shape[1]}")
        self.n_samples_total = self.n_files*self.input_per_file
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        
        # each file must have same number of files, otherwise it will be wrong
        print(f"find files at location {self.location}. Number of examples total: {self.n_samples_per_file}. To-use data per trajectory: {self.input_per_file}  Image Shape: {self.img_shape_x} x {self.img_shape_y} x {self.n_in_channels} LR Image Shape: {self.img_shape_x_lr} x {self.img_shape_y_lr} x {self.n_in_channels}")

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['tasks']
        # self.times[file_idx] = _file['scales/sim_time'] disabeld and add to the model with aug convolution

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        y_list = []
        t_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # lr 
        if local_idx !=self.idx_matrix[global_idx%self.input_per_file][0]:
                raise ValueError(f"Invalid Input index: {local_idx} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        if self.n_in_channels ==3:
            w,u,v = self.files[file_idx]["vorticity"][local_idx],self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            w_lr,u_lr,v_lr = self.files[file_idx]["vorticity_lr"][local_idx],self.files[file_idx]["u_lr"][local_idx],self.files[file_idx]["v_lr"][local_idx]
            w,u,v = self.transform(w),self.transform(u),self.transform(v)
            w_lr,u_lr,v_lr = self.transform(w_lr),self.transform(u_lr),self.transform(v_lr)
            y = torch.stack((w,u,v),dim = 0)
            X = torch.stack((w_lr,u_lr,v_lr),dim = 0)
        elif self.n_in_channels ==2:
            u,v = self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            u_lr,v_lr = self.files[file_idx]["u_lr"][local_idx],self.files[file_idx]["v_lr"][local_idx]
            u,v = self.transform(u),self.transform(v)
            u_lr,v_lr = self.transform(u_lr),self.transform(v_lr)
            y = torch.stack((u,v),dim = 0)
            X = torch.stack((u_lr,v_lr),dim = 0)
        elif self.n_in_channels ==1:
            w = self.files[file_idx]["vorticity"][local_idx]
            w_lr = self.files[file_idx]["vorticity_lr"][local_idx]
            w = self.transform(w)
            w_lr = self.transform(w_lr)
            y = w.unsqueeze(0)
            X = w_lr.unsqueeze(0)
        y_list.append(y)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i*self.timescale_factor
            if local_idx_future !=self.idx_matrix[global_idx%self.input_per_file][i]:
                raise ValueError(f"Invalid target index: {local_idx_future} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
            
            if self.n_in_channels ==3:
                w,u,v = self.files[file_idx]["vorticity"][local_idx_future],self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                w,u,v = self.transform(w),self.transform(u),self.transform(v)
                y = torch.stack((w,u,v),dim = 0)
            elif self.n_in_channels ==2:
                u,v = self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                u,v = self.transform(u),self.transform(v)
                y = torch.stack((u,v),dim = 0)
            elif self.n_in_channels ==1:
                w = self.files[file_idx]["vorticity"][local_idx_future]
                w = self.transform(w)
                y = w.unsqueeze(0)
            y_list.append(y) 
            # t_list.append(t)
        y = torch.stack(y_list,dim = 0) 
        return X,y

    def get_indices(self, global_idx):
        if self.state =="no_roll_out":
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor*self.num_snapshots  # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor  # which sample in that file we are on 
        return file_idx, local_idx
    
    @staticmethod
    def generate_toeplitz(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
        rows = final_index - cols + 2
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix such that it becomes a Toeplitz matrix
        for i in range(rows):
            for j in range(cols):
                value = i + j
                matrix[i, j] = min(value, final_index)
                    
        return matrix
    @staticmethod
    def generate_test_matrix(cols:int, final_index:int):
        # Calculate the number of rows based on the final index and number of columns
        rows = (final_index + 1) // (cols - 1)
        
        # Check if an additional row is needed to reach the final index
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix according to the specified pattern
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                if current_value <= final_index:
                    matrix[i, j] = current_value
                    current_value += 1
            current_value -= 1  # Repeat the last element in the next row
                    
        return matrix[:-1,:]



class GetDataset_diffIC_LowRes_sequence(Dataset):
    '''Dataloader from different initial conditions
    It loads single low-resolution image as a input and gives following high-resolution images as targets
    '''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method,in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        
    def _get_files_stats(self):
        # larger dt = 0.1
        self.files_paths = glob.glob(self.location + "/*.h5") #only take s9
        # /Burger2D_*
        # /rbc_*_256/
        # Decay_turb
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['tasks']["u"].shape[0]
            self.img_shape_x = _f['tasks']["u"].shape[1]
            self.img_shape_y = _f['tasks']["u"].shape[2]
            self.img_shape_x_lr = _f['tasks']["u_lr"].shape[1]
            self.img_shape_y_lr = _f['tasks']["u_lr"].shape[2]
        final_index = (self.n_samples_per_file-1)//self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
            print(self.idx_matrix)
        else:
            self.idx_matrix = self.generate_toeplitz(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        if self.num_snapshots != self.idx_matrix.shape[1] -1:
            raise ValueError(f"Invalid number of snapshots: {self.num_snapshots} vs {self.idx_matrix.shape[1]}")
        self.n_samples_total = self.n_files*self.input_per_file
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        
        # each file must have same number of files, otherwise it will be wrong
        print(f"find files at location {self.location}. Number of examples total: {self.n_samples_per_file}. To-use data per trajectory: {self.input_per_file}  Image Shape: {self.img_shape_x} x {self.img_shape_y} x {self.n_in_channels} LR Image Shape: {self.img_shape_x_lr} x {self.img_shape_y_lr} x {self.n_in_channels}")

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['tasks']
        # self.times[file_idx] = _file['scales/sim_time']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        X_list = []
        y_list = []
        t_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # lr 
        if local_idx !=self.idx_matrix[global_idx%self.input_per_file][0]:
                raise ValueError(f"Invalid Input index: {local_idx} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        if self.n_in_channels ==3:
            w,u,v = self.files[file_idx]["vorticity"][local_idx],self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            w_lr,u_lr,v_lr = self.files[file_idx]["vorticity_lr"][local_idx],self.files[file_idx]["u_lr"][local_idx],self.files[file_idx]["v_lr"][local_idx]
            w,u,v = self.transform(w),self.transform(u),self.transform(v)
            w_lr,u_lr,v_lr = self.transform(w_lr),self.transform(u_lr),self.transform(v_lr)
            y = torch.stack((w,u,v),dim = 0)
            X = torch.stack((w_lr,u_lr,v_lr),dim = 0)
        elif self.n_in_channels ==2:
            u,v = self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            u_lr,v_lr = self.files[file_idx]["u_lr"][local_idx],self.files[file_idx]["v_lr"][local_idx]
            u,v = self.transform(u),self.transform(v)
            u_lr,v_lr = self.transform(u_lr),self.transform(v_lr)
            y = torch.stack((u,v),dim = 0)
            X = torch.stack((u_lr,v_lr),dim = 0)
        elif self.n_in_channels ==1:
            w = self.files[file_idx]["vorticity"][local_idx]
            w_lr = self.files[file_idx]["vorticity_lr"][local_idx]
            w = self.transform(w)
            w_lr = self.transform(w_lr)
            y = w.unsqueeze(0)
            X = w_lr.unsqueeze(0)
        X_list.append(X)
        y_list.append(y)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i*self.timescale_factor
            if local_idx_future !=self.idx_matrix[global_idx%self.input_per_file][i]:
                raise ValueError(f"Invalid target index: {local_idx_future} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
            
            if self.n_in_channels ==3:
                w,u,v = self.files[file_idx]["vorticity"][local_idx_future],self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                w,u,v = self.transform(w),self.transform(u),self.transform(v)
                y = torch.stack((w,u,v),dim = 0)
            elif self.n_in_channels ==2:
                u,v = self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                u,v = self.transform(u),self.transform(v)
                y = torch.stack((u,v),dim = 0)
            elif self.n_in_channels ==1:
                w = self.files[file_idx]["vorticity"][local_idx_future]
                w = self.transform(w)
                y = w.unsqueeze(0)
            y_list.append(y) 
            # t_list.append(t)
        if self.n_in_channels ==3:
            w,u,v = self.files[file_idx]["vorticity_lr"][local_idx_future],self.files[file_idx]["u_lr"][local_idx_future],self.files[file_idx]["v_lr"][local_idx_future]
            w,u,v = self.transform(w),self.transform(u),self.transform(v)
            X = torch.stack((w,u,v),dim = 0)
        elif self.n_in_channels ==2:
            u,v = self.files[file_idx]["u_lr"][local_idx_future],self.files[file_idx]["v_lr"][local_idx_future]
            u,v = self.transform(u),self.transform(v)
            X = torch.stack((u,v),dim = 0)
        elif self.n_in_channels ==1:
            w = self.files[file_idx]["vorticity_lr"][local_idx_future]
            w = self.transform(w)
            X = w.unsqueeze(0)
        X_list.append(X)
        y = torch.stack(y_list,dim = 1) 
        X = torch.stack(X_list,dim = 1)
        return X,y

    def get_indices(self, global_idx):
        if self.state =="no_roll_out":
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor*self.num_snapshots  # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor  # which sample in that file we are on 
        return file_idx, local_idx
    
    @staticmethod
    def generate_toeplitz(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
        rows = final_index - cols + 2
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix such that it becomes a Toeplitz matrix
        for i in range(rows):
            for j in range(cols):
                value = i + j
                matrix[i, j] = min(value, final_index)
                    
        return matrix
    @staticmethod
    def generate_test_matrix(cols:int, final_index:int):
        # Calculate the number of rows based on the final index and number of columns
        rows = (final_index + 1) // (cols - 1)
        
        # Check if an additional row is needed to reach the final index
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix according to the specified pattern
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                if current_value <= final_index:
                    matrix[i, j] = current_value
                    current_value += 1
            current_value -= 1  # Repeat the last element in the next row
                    
        return matrix[:-1,:]



class Special_Loader_Fluid(Dataset):
    '''Dataloader for different initial conditions.
    It loads two low-resolution image as inputs and gives all HR snapshots as targets
    '''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method,in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        if method == "bicubic":
            self.input_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False) # TODO: compatibility issue for antialias='warn' check torch version
        elif method == "gaussian_blur":
            self.input_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))
        
    def _get_files_stats(self):
        # larger dt = 0.1
        self.files_paths = glob.glob(self.location + "/*.h5") #only take s9
        # /Burger2D_*
        # /rbc_*_256/
        # Decay_turb
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['tasks']["u"].shape[0]
            self.img_shape_x = _f['tasks']["u"].shape[1]
            self.img_shape_y = _f['tasks']["u"].shape[2]

        final_index = (self.n_samples_per_file-1)//self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
            print(self.idx_matrix)
        else:
            self.idx_matrix = self.generate_toeplitz(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        if self.num_snapshots != self.idx_matrix.shape[1] -1:
            raise ValueError(f"Invalid number of snapshots: {self.num_snapshots} vs {self.idx_matrix.shape[1]}")
        self.n_samples_total = self.n_files*self.input_per_file
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        
        # each file must have same number of files, otherwise it will be wrong
        print("Found data at path {}. Number of examples total: {}. To-use data per trajectory: {}  Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_per_file, self.input_per_file,self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['tasks']
        # self.times[file_idx] = _file['scales/sim_time']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        y_list = []
        t_list = []
        X_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # lr 
        if local_idx !=self.idx_matrix[global_idx%self.input_per_file][0]:
                raise ValueError(f"Invalid Input index: {local_idx} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        if self.n_in_channels ==3:
            w,u,v = self.files[file_idx]["vorticity"][local_idx],self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            w,u,v = self.transform(w),self.transform(u),self.transform(v)
            y = torch.stack((w,u,v),dim = 0)
        elif self.n_in_channels ==2:
            u,v = self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            u,v = self.transform(u),self.transform(v)
            y = torch.stack((u,v),dim = 0)
        elif self.n_in_channels ==1:
            w = self.files[file_idx]["vorticity"][local_idx]
            w = self.transform(w)
            y = w.unsqueeze(0)
        X = self.get_X(y)
        y_list.append(y)
        X_list.append(X)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i*self.timescale_factor
            if local_idx_future !=self.idx_matrix[global_idx%self.input_per_file][i]:
                raise ValueError(f"Invalid target index: {local_idx_future} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
            
            if self.n_in_channels ==3:
                w,u,v = self.files[file_idx]["vorticity"][local_idx_future],self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                w,u,v = self.transform(w),self.transform(u),self.transform(v)
                y = torch.stack((w,u,v),dim = 0)
            elif self.n_in_channels ==2:
                u,v = self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                u,v = self.transform(u),self.transform(v)
                y = torch.stack((u,v),dim = 0)
            elif self.n_in_channels ==1:
                w = self.files[file_idx]["vorticity"][local_idx_future]
                w = self.transform(w)
                y = w.unsqueeze(0)
            y_list.append(y) 
        X_list.append(self.get_X(y)) # first one and last one as LR input
            # t_list.append(t)
        y = torch.stack(y_list,dim = 1)
        X = torch.stack(X_list,dim = 1) 
        # t = torch.stack(t_list,dim = 0) 
        return X,y

    def get_indices(self, global_idx):
        if self.state =="no_roll_out":
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor*self.num_snapshots  # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor  # which sample in that file we are on 
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
    
    @staticmethod
    def generate_toeplitz(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
        rows = final_index - cols + 2
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix such that it becomes a Toeplitz matrix
        for i in range(rows):
            for j in range(cols):
                value = i + j
                matrix[i, j] = min(value, final_index)
                    
        return matrix
    @staticmethod
    def generate_test_matrix(cols:int, final_index:int):
        # Calculate the number of rows based on the final index and number of columns
        rows = (final_index + 1) // (cols - 1)
        
        # Check if an additional row is needed to reach the final index
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix according to the specified pattern
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                if current_value <= final_index:
                    matrix[i, j] = current_value
                    current_value += 1
            current_value -= 1  # Repeat the last element in the next row
                    
        return matrix[:-1,:]


class FNO_Special_Loader_Fluid(Dataset):
    '''Dataloader for different initial conditions.
    It loads intial condition w0,u0,v0 and coordinates as inputs and gives all HR snapshots as targets.
    Train FNO as presented in their paper, zero-shot super-resolution setting.
    '''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method,in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        if method == "bicubic":
            self.input_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False) # TODO: compatibility issue for antialias='warn' check torch version
            self.upsample = transforms.Resize((self.img_shape_x,self.img_shape_y),Image.BICUBIC,antialias=False)
        elif method == "gaussian_blur":
            self.input_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))
        
    def _get_files_stats(self):
        # larger dt = 0.1
        self.files_paths = glob.glob(self.location + "/*.h5") #only take s9
        # /Burger2D_*
        # /rbc_*_256/
        # Decay_turb
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['tasks']["u"].shape[0]
            self.img_shape_x = _f['tasks']["u"].shape[1]
            self.img_shape_y = _f['tasks']["u"].shape[2]

        final_index = (self.n_samples_per_file-1)//self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
            print(self.idx_matrix)
        else:
            self.idx_matrix = self.generate_toeplitz(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        if self.num_snapshots != self.idx_matrix.shape[1] -1:
            raise ValueError(f"Invalid number of snapshots: {self.num_snapshots} vs {self.idx_matrix.shape[1]}")
        self.n_samples_total = self.n_files*self.input_per_file
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        
        # each file must have same number of files, otherwise it will be wrong
        print("Found data at path {}. Number of examples total: {}. To-use data per trajectory: {}  Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_per_file, self.input_per_file,self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['tasks']
        # self.times[file_idx] = _file['scales/sim_time']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        y_list = []
        t_list = []
        X_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # load grid to B,C,T,X,Y
        gridx,gridy,gridt = torch.tensor(np.linspace(0,1,self.img_shape_x+1)[:-1]),torch.tensor(np.linspace(0,1,self.img_shape_y+1)[:-1]),torch.tensor(np.linspace(0,1,self.num_snapshots+1))
        gridx = gridx.reshape(1,1,self.img_shape_x,1).repeat([1,self.num_snapshots+1,1,self.img_shape_y])
        gridy = gridy.reshape(1,1,1,self.img_shape_y).repeat([1,self.num_snapshots+1,self.img_shape_x,1])
        gridt = gridt.reshape(1,self.num_snapshots+1,1,1).repeat([1,1,self.img_shape_x,self.img_shape_y])
        if local_idx !=self.idx_matrix[global_idx%self.input_per_file][0]:
                raise ValueError(f"Invalid Input index: {local_idx} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        if self.n_in_channels ==3:
            w,u,v = self.files[file_idx]["vorticity"][local_idx],self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            w,u,v = self.transform(w),self.transform(u),self.transform(v)
            y = torch.stack((w,u,v),dim = 0)
        elif self.n_in_channels ==2:
            u,v = self.files[file_idx]["u"][local_idx],self.files[file_idx]["v"][local_idx]
            u,v = self.transform(u),self.transform(v)
            y = torch.stack((u,v),dim = 0)
        elif self.n_in_channels ==1:
            w = self.files[file_idx]["vorticity"][local_idx]
            w = self.transform(w)
            y = w.unsqueeze(0)
        X_LR = self.get_X(y)
        X_interp = self.upsample(X_LR)
        X_interp = X_interp.reshape(self.n_in_channels,1,self.img_shape_x,self.img_shape_y).repeat([1, self.num_snapshots+1, 1,1])
        X = torch.cat((X_interp,gridx,gridy,gridt),dim = 0)
        y_list.append(y)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i*self.timescale_factor
            if local_idx_future !=self.idx_matrix[global_idx%self.input_per_file][i]:
                raise ValueError(f"Invalid target index: {local_idx_future} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
            
            if self.n_in_channels ==3:
                w,u,v = self.files[file_idx]["vorticity"][local_idx_future],self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                w,u,v = self.transform(w),self.transform(u),self.transform(v)
                y = torch.stack((w,u,v),dim = 0)
            elif self.n_in_channels ==2:
                u,v = self.files[file_idx]["u"][local_idx_future],self.files[file_idx]["v"][local_idx_future]
                u,v = self.transform(u),self.transform(v)
                y = torch.stack((u,v),dim = 0)
            elif self.n_in_channels ==1:
                w = self.files[file_idx]["vorticity"][local_idx_future]
                w = self.transform(w)
                y = w.unsqueeze(0)
            y_list.append(y) 
            # t_list.append(t)
        y = torch.stack(y_list,dim = 1)
        # t = torch.stack(t_list,dim = 0) 
        return X,y

    def get_indices(self, global_idx):
        if self.state =="no_roll_out":
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor *self.num_snapshots # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor  # which sample in that file we are on 
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
    
    @staticmethod
    def generate_toeplitz(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
        rows = final_index - cols + 2
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix such that it becomes a Toeplitz matrix
        for i in range(rows):
            for j in range(cols):
                value = i + j
                matrix[i, j] = min(value, final_index)
                    
        return matrix
    @staticmethod
    def generate_test_matrix(cols:int, final_index:int):
        # Calculate the number of rows based on the final index and number of columns
        rows = (final_index + 1) // (cols - 1)
        
        # Check if an additional row is needed to reach the final index
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix according to the specified pattern
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                if current_value <= final_index:
                    matrix[i, j] = current_value
                    current_value += 1
            current_value -= 1  # Repeat the last element in the next row
                    
        return matrix[:-1,:]
    

# 
class GetClimateDatasets(Dataset):
    '''Dataloader from different initial conditions
    It loads single low-resolution image as a input and gives following high-resolution images as targets
    '''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method,in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        self.basic_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))
        if method == "bicubic":
            self.input_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False) # TODO: compatibility issue for antialias='warn' check torch version
        
    def _get_files_stats(self):
        # larger dt = 0.1
        self.files_paths = glob.glob(self.location + "/*.h5") #only take s9
        # /Burger2D_*
        # /rbc_*_256/
        # Decay_turb
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['fields'].shape[0]
            self.img_shape_x = _f['fields'].shape[1]
            self.img_shape_y = _f['fields'].shape[2]

        final_index = (self.n_samples_per_file-1)//self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
            print(self.idx_matrix)
        else:
            self.idx_matrix = self.generate_toeplitz(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        if self.num_snapshots != self.idx_matrix.shape[1] -1:
            raise ValueError(f"Invalid number of snapshots: {self.num_snapshots} vs {self.idx_matrix.shape[1]}")
        self.n_samples_total = self.n_files*self.input_per_file
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        
        # each file must have same number of files, otherwise it will be wrong
        print("Found data at path {}. Number of examples total: {}. To-use data per trajectory: {}  Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_per_file, self.input_per_file,self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['fields']
        # self.times[file_idx] = _file['scales/sim_time']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        y_list = []
        t_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # lr 
        if local_idx !=self.idx_matrix[global_idx%self.input_per_file][0]:
                raise ValueError(f"Invalid Input index: {local_idx} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
        if self.files[file_idx] is None:
            self._open_file(file_idx)
        w = self.files[file_idx][local_idx]
        w = self.transform(w)
        y = w.unsqueeze(0)
        X = self.get_X(y)
        y_list.append(y)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i*self.timescale_factor
            if local_idx_future !=self.idx_matrix[global_idx%self.input_per_file][i]:
                raise ValueError(f"Invalid target index: {local_idx_future} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
            w = self.files[file_idx][local_idx_future]
            w = self.transform(w)
            y = w.unsqueeze(0)
            y_list.append(y) 
            # t_list.append(t)
        y = torch.stack(y_list,dim = 0) 
        # t = torch.stack(t_list,dim = 0) 
        return X,y

    def get_indices(self, global_idx):
        if self.state =="no_roll_out":
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor *self.num_snapshots  # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor # which sample in that file we are on 
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
    
    @staticmethod
    def generate_toeplitz(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
        rows = final_index - cols + 2
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix such that it becomes a Toeplitz matrix
        for i in range(rows):
            for j in range(cols):
                value = i + j
                matrix[i, j] = min(value, final_index)
                    
        return matrix
    @staticmethod
    def generate_test_matrix(cols:int, final_index:int):
        # Calculate the number of rows based on the final index and number of columns
        rows = (final_index + 1) // (cols - 1)
        
        # Check if an additional row is needed to reach the final index
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix according to the specified pattern
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                if current_value <= final_index:
                    matrix[i, j] = current_value
                    current_value += 1
            current_value -= 1  # Repeat the last element in the next row
                    
        return matrix[:-1,:]


class GetClimateDatasets_special(Dataset):
    '''Dataloader from different initial conditions
    It loads single low-resolution image as a input and gives following high-resolution images as targets
    '''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method,in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        self.basic_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))
        if method == "bicubic":
            self.input_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False) # TODO: compatibility issue for antialias='warn' check torch version∂
        
    def _get_files_stats(self):
        # larger dt = 0.1
        self.files_paths = glob.glob(self.location + "/*.h5") #only take s9
        # /Burger2D_*
        # /rbc_*_256/
        # Decay_turb
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['fields'].shape[0]
            self.img_shape_x = _f['fields'].shape[1]
            self.img_shape_y = _f['fields'].shape[2]

        final_index = (self.n_samples_per_file-1)//self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
            print(self.idx_matrix)
        else:
            self.idx_matrix = self.generate_toeplitz(cols = self.num_snapshots+1, final_index=final_index)*self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        if self.num_snapshots != self.idx_matrix.shape[1] -1:
            raise ValueError(f"Invalid number of snapshots: {self.num_snapshots} vs {self.idx_matrix.shape[1]}")
        self.n_samples_total = self.n_files*self.input_per_file
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        
        # each file must have same number of files, otherwise it will be wrong
        print("Found data at path {}. Number of examples total: {}. To-use data per trajectory: {}  Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_per_file, self.input_per_file,self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['fields']
        # self.times[file_idx] = _file['scales/sim_time']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        y_list = []
        t_list = []
        X_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # lr 
        if local_idx !=self.idx_matrix[global_idx%self.input_per_file][0]:
                raise ValueError(f"Invalid Input index: {local_idx} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
        if self.files[file_idx] is None:
            self._open_file(file_idx)
        w = self.files[file_idx][local_idx]
        w = self.transform(w)
        y = w.unsqueeze(0)
        X = self.get_X(y)
        X_list.append(X)
        y_list.append(y)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i*self.timescale_factor
            if local_idx_future !=self.idx_matrix[global_idx%self.input_per_file][i]:
                raise ValueError(f"Invalid target index: {local_idx_future} vs index matrix {self.idx_matrix[global_idx%self.input_per_file][0]}")
            w = self.files[file_idx][local_idx_future]
            w = self.transform(w)
            y = w.unsqueeze(0)
            y_list.append(y) 
            # t_list.append(t)
        X_list.append(self.get_X(y)) # first one and last one as LR input
        y = torch.stack(y_list,dim = 1) 
        X = torch.stack(X_list,dim = 1)
        # t = torch.stack(t_list,dim = 0) 
        return X,y

    def get_indices(self, global_idx):
        if self.state =="no_roll_out":
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor *self.num_snapshots  # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.input_per_file)  # which file we are on
            local_idx = int(global_idx % self.input_per_file) * self.timescale_factor # which sample in that file we are on 
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
    
    @staticmethod
    def generate_toeplitz(cols:int, final_index:int):
    # Calculate the number of rows based on the final index and number of columns
        rows = final_index - cols + 2
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix such that it becomes a Toeplitz matrix
        for i in range(rows):
            for j in range(cols):
                value = i + j
                matrix[i, j] = min(value, final_index)
                    
        return matrix
    @staticmethod
    def generate_test_matrix(cols:int, final_index:int):
        # Calculate the number of rows based on the final index and number of columns
        rows = (final_index + 1) // (cols - 1)
        
        # Check if an additional row is needed to reach the final index
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        
        # Initialize a matrix filled with zeros
        matrix = np.zeros((rows, cols))
        
        # Fill the matrix according to the specified pattern
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                if current_value <= final_index:
                    matrix[i, j] = current_value
                    current_value += 1
            current_value -= 1  # Repeat the last element in the next row
                    
        return matrix[:-1,:]


def random_split(dataset, lengths,
                 generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.

    If a list of fractions that sum up to 1 is given,
    the lengths will be computed automatically as
    floor(frac * len(dataset)) for each fraction provided.

    After computing the lengths, if there are any remainders, 1 count will be
    distributed in round-robin fashion to the lengths
    until there are no remainders left.

    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))
    >>> random_split(range(30), [0.3, 0.3, 0.4], generator=torch.Generator(
    ...   ).manual_seed(42))

    Args:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths or fractions of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(f"Length of split at index {i} is 0. "
                              f"This might result in an empty dataset.")

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()  # type: ignore[call-overload]
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]



if __name__ == "__main__":

    train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(data_name= 'decay_turb',batch_size= 30,data_path="/pscratch/sd/j/junyi012/Decay_Turbulence_small",in_channels =3,timescale_factor= 20)
    for idx, (input,target) in enumerate (test2_loader):
        input = input
        target = target
    print(input.shape)
    print(target.shape)

    print(idx)

    # for i in range(1,6):
    #     plt.figure()
    #     plt.imshow(input[0,0,i,:,:])
    #     plt.savefig(f'debug/input{i}.png')
    #     plt.figure()
    #     plt.imshow(target[0,0,i,:,:])
    #     plt.colorbar()
    #     plt.savefig(f"debug/target{i}.png")

    # list = []
    # for i in range (target.shape[1]):
    #     x = np.linalg.norm(target[:,0,0,:,:]-target[:,i,0,:,:],ord = 2, axis = (1,2))/np.linalg.norm(target[:,i,0,:,:],ord = 2, axis = (1,2))
    #     print(x.mean())
    #     list.append(x.mean())
    # print(list)
    # for idx, (input,target) in enumerate (val1_loader):
    #     input = input
    #     target = target
    # print(input.shape)
    # print(target.shape)
    # for idx, (input,target) in enumerate (test1_loader):
    #     input = input
    #     target = target
    # print(input.shape)
    # print(target.shape)
    # list = []
    # for i in range(2):
    #     x = torch.rand(1,1,128)
    #     list.append(x)
    # x = torch.stack(list,dim = 0)
    # print(x.shape)
