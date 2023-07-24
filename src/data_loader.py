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
import math
from torch import default_generator, randperm
from torch._utils import _accumulate
from torch.utils.data.dataset import Subset
#"../superbench/datasets/nskt16000_1024"
#../datasets/rbc_diff_IC/rbc_IC1
def getData(data_name = "rbc_diff_IC", data_path =  "../rbc_diff_IC/rbc_10IC",
             upscale_factor = 4,timescale_factor = 4, num_snapshots = 20,
             noise_ratio = 0.0, crop_size = 512, method = "bicubic", 
             batch_size = 1, std = [0.6703, 0.6344, 8.3615]):  
    '''
    Loading data from four dataset folders: (a) nskt_16k; (b) nskt_32k; (c) cosmo; (d) era5.
    Each dataset contains: 
        - 1 train dataset, 
        - 2 validation sets (interpolation and extrapolation), 
        - 2 test sets (interpolation and extrapolation),
        
    ===
    std: the channel-wise standard deviation of each dataset, list: [#channels]
    
    '''
    if data_name == "rbc_diff_IC_3test":
        #To do swap and change 
        train_loader = get_data_loader(data_name, data_path, '/rbc_IC1', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std)
        val1_loader = get_data_loader(data_name, data_path, '/rbc_IC1', "val", upscale_factor, timescale_factor//2,num_snapshots*2,noise_ratio, crop_size, method, batch_size, std)
        val2_loader = get_data_loader(data_name, data_path, '/rbc_IC1', "val", upscale_factor,timescale_factor//4,num_snapshots*4,noise_ratio, crop_size, method, batch_size, std) 
        test3_loader = get_data_loader(data_name, data_path, '/rbc_IC2', "test", upscale_factor,timescale_factor, num_snapshots,noise_ratio, crop_size, method, batch_size, std)
        test1_loader = get_data_loader(data_name, data_path, '/rbc_IC2', "test", upscale_factor,timescale_factor//2, num_snapshots*2, noise_ratio, crop_size, method, batch_size, std)
        test2_loader = get_data_loader(data_name, data_path, '/rbc_IC2', "test", upscale_factor,timescale_factor//4, num_snapshots*4, noise_ratio, crop_size, method, batch_size, std)
        return train_loader, val1_loader, val2_loader, test1_loader, test2_loader,test3_loader

    elif data_name == "rbc_diff_IC":
        #To do swap and change 
        train_loader = get_data_loader(data_name, data_path, '/rbc_IC1', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std)
        val1_loader = get_data_loader(data_name, data_path, '/rbc_IC1', "val", upscale_factor, timescale_factor//2,4,noise_ratio, crop_size, method, batch_size, std)
        val2_loader = get_data_loader(data_name, data_path, '/rbc_IC2', "val", upscale_factor,timescale_factor//2,4,noise_ratio, crop_size, method, batch_size, std) 
        test1_loader = get_data_loader(data_name, data_path, '/rbc_IC1', "test", upscale_factor,timescale_factor//4, num_snapshots*4, noise_ratio, crop_size, method, batch_size, std)
        test2_loader = get_data_loader(data_name, data_path, '/rbc_IC2', "test", upscale_factor,timescale_factor//4, num_snapshots*4, noise_ratio, crop_size, method, batch_size, std)
        return train_loader, val1_loader, val2_loader, test1_loader, test2_loader
    elif data_name == "rbc_diff_10IC":
        #To do swap and change 
        train_loader = get_data_loader(data_name, data_path, '/train', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std)
        val1_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor, timescale_factor//2,4,noise_ratio, crop_size, method, batch_size, std)
        val2_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor//2,4,noise_ratio, crop_size, method, batch_size, std) 
        test1_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor//4,num_snapshots*4, noise_ratio, crop_size, method, batch_size, std)
        test2_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor//4, num_snapshots*4, noise_ratio, crop_size, method, batch_size, std)
        return train_loader,val1_loader,val2_loader,test1_loader,test2_loader
    elif data_name == "nskt_16k":
        train_loader = get_data_loader(data_name, data_path, '/train', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std)
        val1_loader = get_data_loader(data_name, data_path, '/train', "val", upscale_factor, timescale_factor//2,4,noise_ratio, crop_size, method, batch_size, std)
        val2_loader = get_data_loader(data_name, data_path, '/valid_1', "val", upscale_factor,timescale_factor//2,4,noise_ratio, crop_size, method, batch_size, std) 
        test1_loader = get_data_loader(data_name, data_path, '/train', "test", upscale_factor,timescale_factor//4, num_snapshots*4,noise_ratio, crop_size, method, batch_size, std)
        test2_loader = get_data_loader(data_name, data_path, '/valid_1', "test", upscale_factor,timescale_factor//4, num_snapshots*4, noise_ratio, crop_size, method, batch_size, std)
    # val1_loader, val2_loader, test1_loader, test2_loader  = 0,0,0,0
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
                                num_workers = 4, # TODO: make a param
                                shuffle = shuffle, 
                                sampler = None,
                                drop_last = True,
                                pin_memory = False)

        return dataloader
    
    elif data_name in ['rbc_diff_IC']:
        dataset = GetRBCDataset(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method) 
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
          
    elif data_name in ['rbc_diff_10IC']:
        dataset = GetRBCDataset_diff_IC(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method) 
        if state == "train":
            shuffle = False
        else:
            shuffle = False

        dataloader = DataLoader(dataset,
                                batch_size = int(batch_size),
                                num_workers = 0, # TODO: make a param
                                shuffle = shuffle, 
                                sampler = None,
                                drop_last = True,
                                pin_memory = False)
        return dataloader    
    # if data_name in ['nskt_16k']:
    #     dataset = GetFluidDataset(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method) 
    # elif data_name in ['RBC']:
    #     dataset = GetRBCDataset(data_path+data_tag, state, transform, upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method) 
    #     trianset,val1set = random_split(dataset,[0.9,0.1],generator=torch.Generator().manual_seed(args.seed))
    #     testset,val2set = random_split(dataset,[0.9,0.1],generator=torch.Generator().manual_seed(args.seed))
    # if state == "train":
    #     shuffle = False
        
    # else:
    #     shuffle = False




class GetRBCDataset(Dataset):
    '''Dataloader class for NSKT and cosmo datasets'''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method):
        self.location = location
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self._get_files_stats()
        self.crop_size = crop_size
        self.crop_transform = transforms.CenterCrop(crop_size)
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        if method == "bicubic":
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                                             transforms.Resize((int(self.crop_size/upscale_factor),int(self.crop_size/upscale_factor)),Image.BICUBIC) ]) # TODO: compatibility issue for antialias='warn' check torch version
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
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['tasks']["vorticity"].shape[0]
            self.n_in_channels = 1
            self.img_shape_x = _f['tasks']["vorticity"].shape[1]
            self.img_shape_y = _f['tasks']["vorticity"].shape[2]
        with h5py.File(self.files_paths[-1], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[-1]))
            n_samples_last_file = _f['tasks']["vorticity"].shape[0]
        self.n_samples_total = self.n_samples_per_file*(self.n_files - 1)+ n_samples_last_file
        self.files = [None for _ in range(self.n_files)]
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['tasks']['vorticity']

    def __len__(self):
        return self.n_samples_total-self.timescale_factor*self.num_snapshots-1

    def __getitem__(self, global_idx):
        y_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # print(file_idx)
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        y = self.transform(self.files[file_idx][local_idx]) # from numpy to torch
        y = self.target_transform(y).unsqueeze(0) # cropping the image
        X = self.get_X(y) # getting the input
        # getting the future samples
        y_list.append(y)
        for i in range(1, self.num_snapshots+1):
            file_idx, local_idx_future = self.get_indices(global_idx + i*self.timescale_factor)
            #open image file for future sample
            if self.files[file_idx] is None:
                self._open_file(file_idx)
            y = self.transform(self.files[file_idx][local_idx_future])
            y = self.target_transform(y).unsqueeze(0)
            y_list.append(y)
        y = torch.stack(y_list,dim = 0) 
        return X,y

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

class GetRBCDataset_diff_IC(Dataset):
    '''Dataloader class for NSKT and cosmo datasets'''
    def __init__(self, location, state, transform, upscale_factor,timescale_factor,num_snapshots, noise_ratio, std,crop_size, method):
        self.location = location
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self.n_samples_total = 0
        self.n_samples_per_file = 0
        self.n_in_channels = 1
        self.img_shape_x = 0
        self.img_shape_y = 0
        self.crop_size = crop_size
        self.crop_transform = transforms.CenterCrop(crop_size)
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        if method == "bicubic":
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                                             transforms.Resize((int(self.crop_size/upscale_factor),int(self.crop_size/upscale_factor)),Image.BICUBIC) ]) # TODO: compatibility issue for antialias='warn' check torch version
        elif method == "gaussian_blur":
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
                                                       transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))])
        elif method == "uniform":
            self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
                                        ])
        self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size) # since it's the target, we keep its original quality
                                        ])
        self._get_files_stats()
    def _get_files_stats(self):
        # location in raocp rbc_10IC
        self.files_paths = glob.glob(self.location + "/rbc_*_256/rbc_*_256_s9.h5") #only take s9
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        print("Found {} files".format(self.n_files))
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['tasks']["vorticity"].shape[0]
            self.n_in_channels = 1
            self.img_shape_x = _f['tasks']["vorticity"].shape[1]
            self.img_shape_y = _f['tasks']["vorticity"].shape[2]
        self.number_input = self.n_samples_per_file - self.timescale_factor*self.num_snapshots-1
        print(self.number_input)
        self.n_samples_total = self.n_files*self.number_input
        # change correspond to data structure
        self.files = [None for _ in range(self.n_files)]
        self.times = [None for _ in range(self.n_files)]
        # each file must have same number of files, otherwise it will be wrong
        print(self.n_samples_total)
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_per_file, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['tasks']['vorticity']
        self.times[file_idx] = _file['scales/sim_time']

    def __len__(self):
        return self.n_samples_total-1

    def __getitem__(self, global_idx):
        y_list = []
        t_list = []
        file_idx, local_idx = self.get_indices(global_idx)
        # lr 
        if self.files[file_idx] is None:
                self._open_file(file_idx)
        y = self.transform(self.files[file_idx][local_idx]) # from numpy to torch
        y = self.target_transform(y).unsqueeze(0) # cropping the image
        X = self.get_X(y) # getting the input
        t = self.transform(np.array(self.times[file_idx][local_idx]))
        t_list.append(t)
        y_list.append(y)
        # getting the future samples
        for i in range(1, self.num_snapshots+1):
            local_idx_future = local_idx+i
            y = self.transform(self.files[file_idx][local_idx_future])
            t = self.transform(np.array(self.times[file_idx][local_idx_future]))
            y = self.target_transform(y).unsqueeze(0)
            y_list.append(y)
            t_list.append(t)
        y = torch.stack(y_list,dim = 0) 
        t = torch.stack(t_list,dim = 0) 
        return X,y

    def get_indices(self, global_idx):
        file_idx = int(global_idx/self.number_input)  # which file we are on
        local_idx = int(global_idx % self.number_input)  # which sample in that file we are on 

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
    train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(data_name= 'rbc_diff_10IC',batch_size= 1,crop_size = 128)
    for idx, (input,target) in enumerate (test2_loader):
        input = input
        target = target
    print(input.shape)
    print(target.shape)
    print(idx)
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
