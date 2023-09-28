import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import h5py
import torchvision.transforms as transforms
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
    if data_name in["rbc_small", "Burger2D_small", "Decay_turb_small"] or  "FNO" in data_name or "ConvLSTM" in data_name:
        #To do swap and change 
        if timescale_factor > 1:
            train_loader = get_data_loader(data_name, data_path, '/train', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels)
            val1_loader = get_data_loader(data_name, data_path, '/val', "test", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels)
            val2_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std, in_channels)
            test1_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor,num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels)
            test2_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor, num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels)
        else: 
            train_loader = get_data_loader(data_name, data_path, '/train', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels)
            val1_loader = get_data_loader(data_name, data_path, '/val', "val", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels)
            val2_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std, in_channels)
            test1_loader = get_data_loader(data_name, data_path, '/test', "test_one", upscale_factor,timescale_factor,num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels)
            test2_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor, num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels)    
        return train_loader,val1_loader,val2_loader,test1_loader,test2_loader
    
def get_data_loader(data_name, data_path, data_tag, state, upscale_factor, timescale_factor, num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels=1):
    
    transform = torch.from_numpy
    if "FNO" or "ConvLSTM" in data_name:
        dataset = Special_Loader(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels) 
    else:
        dataset = GetDataset_diffIC_NOCrop(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels) 

    if state == "train":
        shuffle = True
    else:
        shuffle = False

    dataloader = DataLoader(dataset,
                            batch_size = int(batch_size),
                            num_workers = 4, # TODO: make a param
                            shuffle = shuffle, 
                            sampler = None,
                            drop_last = True,
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
        if self.state == "test_one":
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



class Special_Loader(Dataset):
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
        if self.state == "test_one":
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

# n_snapshot = 10

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

    train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(data_name= 'rbc_small',batch_size= 30,data_path="../RBC_small",in_channels=1,timescale_factor= 10)
    for idx, (input,target) in enumerate (test1_loader):
        input = input
        target = target
    print(input.shape)
    print(target.shape)
    print(idx)
    plt.figure()
    plt.imshow(input[0,0,:,:])
    plt.savefig('input.png')
    for i in range(1,6):
        plt.figure()
        plt.imshow(target[0,i,0,:,:])
        plt.colorbar()
        plt.savefig(f"target{i}.png")

    list = []
    for i in range (target.shape[1]):
        x = np.linalg.norm(target[:,0,0,:,:]-target[:,i,0,:,:],ord = 2, axis = (1,2))/np.linalg.norm(target[:,i,0,:,:],ord = 2, axis = (1,2))
        print(x.mean())
        list.append(x.mean())
    print(list)
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
