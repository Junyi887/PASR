import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import glob
from torchvision import transforms
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

def getData_test(data_name = "rbc_diff_IC", data_path =  "../rbc_diff_IC/rbc_10IC",
             upscale_factor= 4,timescale_factor = 1, num_snapshots = 20,
             noise_ratio = 0.0, crop_size = 128, method = "bicubic", 
             batch_size = 1, std = [0.6703, 0.6344, 8.3615],in_channels = 1):
    if data_name == "lrsim":
        test_dataset = GetDataset_lres_viz(data_path+"/test", "no_roll_out",torch.from_numpy , upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,sampler = None,drop_last = False,pin_memory = False)
        test_dataset_viz = GetDataset_lres_viz(data_path+"/viz", "no_roll_out",torch.from_numpy , upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
        test_loader_viz = DataLoader(test_dataset_viz,batch_size=512,shuffle=False,sampler = None,drop_last = False,pin_memory = False)
        test_dataset_viz2 = GetDataset_lres_viz(data_path+"/viz", "roll_out",torch.from_numpy , upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
        test_loader_viz2 = DataLoader(test_dataset_viz2,batch_size=512,shuffle=False,sampler = None,drop_last = False,pin_memory = False)
        return test_loader,test_loader_viz,test_loader_viz2
    else:
        test1_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor,num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels,shuffle=False,drop_last=False)
        test2_loader = get_data_loader(data_name, data_path, '/test', "no_roll_out", upscale_factor,timescale_factor, num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels,shuffle=False,drop_last=False)
        test3_loader = get_data_loader(data_name, data_path, '/viz', "no_roll_out", upscale_factor,timescale_factor, num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels,shuffle=False,drop_last=False)
        return test1_loader,test2_loader,test3_loader
    
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
        test1_loader = val2_loader
        test2_loader = val2_loader
        return train_loader,val1_loader,val2_loader,test1_loader,test2_loader
    elif data_name == "climate_sequence":
        dataset = GetClimateDatasets_special(data_path, "train",torch.from_numpy , upscale_factor,timescale_factor, num_snapshots,noise_ratio, std, crop_size, method,in_channels)
        print("Climate Loader")
        train_set,val_set,test_set = random_split(dataset,[0.8,0.1,0.1],generator=torch.Generator().manual_seed(42))
        train_loader = DataLoader(train_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        val1_loader= DataLoader(val_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        val2_loader = DataLoader(test_set,batch_size=batch_size,shuffle=True,sampler = None,drop_last = True,pin_memory = False)
        test1_loader = val2_loader
        test2_loader = val2_loader
        return train_loader,val1_loader,val2_loader,test1_loader,test2_loader
    else:
        train_loader = get_data_loader(data_name, data_path, '/train', "train", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels)
        val1_loader = get_data_loader(data_name, data_path, '/val', "val", upscale_factor, timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels)
        val2_loader = get_data_loader(data_name, data_path, '/test', "no_roll_out", upscale_factor,timescale_factor,num_snapshots,noise_ratio, crop_size, method, batch_size, std, in_channels)
        test1_loader = get_data_loader(data_name, data_path, '/test', "test", upscale_factor,timescale_factor,num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels)
        test2_loader = get_data_loader(data_name, data_path, '/test', "no_roll_out", upscale_factor,timescale_factor, num_snapshots, noise_ratio, crop_size, method, batch_size, std, in_channels)
   
        return train_loader,val1_loader,val2_loader,test1_loader,test2_loader
    
def get_data_loader(data_name, data_path, data_tag, state, upscale_factor, timescale_factor, num_snapshots,noise_ratio, crop_size, method, batch_size, std,in_channels=1,shuffle=True,drop_last=True):
    
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
    elif state =="val":
        shuffle = True
        drop_last = drop_last
    else:
        shuffle = False
        drop_last = drop_last
    
    dataloader = DataLoader(dataset,
                            batch_size = int(batch_size),
                            num_workers = 2, # TODO: make a param
                            shuffle = shuffle, 
                            sampler = None,
                            drop_last = drop_last,
                            pin_memory = False)
    return dataloader

class BaseDataset(Dataset):
    def __init__(self, location, state, transform, upscale_factor, timescale_factor, num_snapshots, noise_ratio, std, crop_size, method, in_channels):
        self.location = location
        self.n_in_channels = in_channels
        self.upscale_factor = upscale_factor
        self.state = state
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std), 1, 1)
        self.transform = transform
        self.crop_size = crop_size
        self.method = method
        self.num_snapshots = num_snapshots
        self.timescale_factor = timescale_factor
        self._get_files_stats()
        self.files = [None for _ in range(self.n_files)]
        if method == "bicubic":
            self.input_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False) # TODO: compatibility issue for antialias='warn' check torch version
        elif method == "gaussian_blur":
            self.input_transform = transforms.GaussianBlur(kernel_size=(3,3), sigma=(1,1))

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        if self.n_files == 0:
            raise FileNotFoundError(f"No files found in {self.location}")
        
        with h5py.File(self.files_paths[0], 'r') as f:
            # Assuming a dataset structure, modify as needed
            self.n_samples_per_file = f['tasks']['u'].shape[0]
            self.img_shape_x = f['tasks']['u'].shape[1]
            self.img_shape_y = f['tasks']['u'].shape[2]

        final_index = (self.n_samples_per_file - 1) // self.timescale_factor
        if self.state == "no_roll_out":
            self.idx_matrix = self.generate_test_matrix(cols=self.num_snapshots + 1, final_index=final_index) * self.timescale_factor
        else:
            self.idx_matrix = self.generate_toeplitz(cols=self.num_snapshots + 1, final_index=final_index) * self.timescale_factor
        self.input_per_file = self.idx_matrix.shape[0]
        self.n_samples_total = self.n_files * self.input_per_file
        print(f"find {self.n_files} files at location {self.location}. Number of examples total: {self.n_samples_per_file}. To-use data per trajectory: {self.input_per_file}  Image Shape: {self.img_shape_x} x {self.img_shape_y} x {self.n_in_channels}")

    def _open_file(self, file_idx):
        if self.files[file_idx] is None:
            self.files[file_idx] = h5py.File(self.files_paths[file_idx], 'r')['tasks']

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented in the subclass")

    def get_indices(self, global_idx):
        file_idx = int(global_idx / self.input_per_file)
        if self.state == "no_roll_out":
            local_idx = (global_idx % self.input_per_file) * self.timescale_factor * self.num_snapshots
        else:
            local_idx = (global_idx % self.input_per_file) * self.timescale_factor
        return file_idx, local_idx

    @staticmethod
    def generate_toeplitz(cols, final_index):
        rows = final_index - cols + 2
        matrix = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                matrix[i, j] = min(i + j, final_index)
        return matrix

    @staticmethod
    def generate_test_matrix(cols, final_index):
        rows = (final_index + 1) // (cols - 1)
        if (final_index + 1) % (cols - 1) != 0:
            rows += 1
        matrix = np.zeros((rows, cols))
        current_value = 0
        for i in range(rows):
            for j in range(cols):
                matrix[i, j] = min(current_value, final_index)
                current_value += 1
            current_value -= 1  # Decrement to avoid skipping values
        return matrix[:-1,:]
    
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
    
    def load_sample(self, file_idx, local_idx,channel_names):
        channel_data = []
        for channel in channel_names:
            data = self.files[file_idx][channel][local_idx]
            transformed_data = self.transform(data)
            channel_data.append(transformed_data)

        return torch.stack(channel_data, dim=0) if len(channel_data) > 1 else channel_data[0].unsqueeze(0)

    def load_future_samples(self, file_idx, current_local_idx,channel_names):
        future_samples = []
        for i in range(1, self.num_snapshots + 1):
            future_local_idx = current_local_idx + i * self.timescale_factor
            y_future = self.load_sample(file_idx, future_local_idx,channel_names)
            future_samples.append(y_future)
        return future_samples

    def get_channel_names(self):
        if self.n_in_channels == 3:
            return ["vorticity", "u", "v"]
        elif self.n_in_channels == 2:
            return ["u", "v"]
        elif self.n_in_channels == 1:
            return ["vorticity"]
        else:
            raise ValueError(f"Invalid number of input channels: {self.n_in_channels}")
        
    def get_LR_channel_names(self):
        if self.n_in_channels == 3:
            return ["vorticity_lr", "u_lr", "v_lr"]
        elif self.n_in_channels == 2:
            return ["u_lr", "v_lr"]
        elif self.n_in_channels == 1:
            return ["vorticity_lr"]
        else:
            raise ValueError(f"Invalid number of input channels: {self.n_in_channels}")
    
    def ensure_correct_index(self, global_idx, local_idx):
        expected_index = self.idx_matrix[global_idx % self.input_per_file][0]
        if local_idx != expected_index:
            raise ValueError(f"Invalid Input index: {local_idx} vs expected index {expected_index}")


class FNO_Special_Loader_Fluid(BaseDataset):
    """return B, C, T,H, W"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization specific to this dataset
        self.upsample = transforms.Resize((self.img_shape_x,self.img_shape_y),Image.BICUBIC,antialias=False)
    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx)
        self.ensure_correct_index(global_idx, local_idx)
        gridx,gridy,gridt = torch.tensor(np.linspace(0,1,self.img_shape_x+1)[:-1]),torch.tensor(np.linspace(0,1,self.img_shape_y+1)[:-1]),torch.tensor(np.linspace(0,1,self.num_snapshots+1))
        # gridx = gridx.reshape(1,1,self.img_shape_x,1).repeat([1,self.num_snapshots+1,1,self.img_shape_y])
        # gridy = gridy.reshape(1,1,1,self.img_shape_y).repeat([1,self.num_snapshots+1,self.img_shape_x,1])
        # gridt = gridt.reshape(1,self.num_snapshots+1,1,1).repeat([1,1,self.img_shape_x,self.img_shape_y])

        if self.files[file_idx] is None:
            self._open_file(file_idx)
        channel_names = self.get_channel_names()
        # Load the current sample
        y_current = self.load_sample(file_idx, local_idx,channel_names)
        X_lr = self.get_X(y_current)
        X_interp = self.upsample(X_lr)
        X_interp = X_interp.reshape(self.n_in_channels,1,self.img_shape_x,self.img_shape_y).repeat([1, self.num_snapshots+1, 1,1])
        X = torch.cat([X_interp,gridx,gridy,gridt],dim=0)
        # Load future samples
        y_future = self.load_future_samples(file_idx, local_idx,channel_names)
        # Combine current and future samples
        y_samples = [y_current] + y_future
        y = torch.stack(y_samples, dim=1)
        return X, y


class Special_Loader_Fluid(BaseDataset):
    """return B, C, T,H, W"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization specific to this dataset

    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx)
        self.ensure_correct_index(global_idx, local_idx)

        if self.files[file_idx] is None:
            self._open_file(file_idx)
        channel_names = self.get_channel_names()
        # Load the current sample
        y_current = self.load_sample(file_idx, local_idx,channel_names)
        X_start = self.get_X(y_current)
        # Load future samples
        y_future = self.load_future_samples(file_idx, local_idx,channel_names)
        X_end = self.get_X(y_future[-1])
        # Combine current and future samples
        y_samples = [y_current] + y_future
        X_samples = [X_start] + [X_end]
        y = torch.stack(y_samples, dim=1)
        X = torch.stack(X_samples, dim=1)
        return X, y

class GetDataset_diffIC_NOCrop(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization specific to this dataset

    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx)
        self.ensure_correct_index(global_idx, local_idx)

        if self.files[file_idx] is None:
            self._open_file(file_idx)

        channel_names = self.get_channel_names()
        # Load the current sample
        y_current = self.load_sample(file_idx, local_idx,channel_names)
        X = self.get_X(y_current)

        # Load future samples
        y_future = self.load_future_samples(file_idx, local_idx,channel_names)

        # Combine current and future samples
        y_samples = [y_current] + y_future
        y = torch.stack(y_samples, dim=0)
        return X, y

class GetDataset_diffIC_LowRes_sequence(BaseDataset):
    '''return B,C, T, H, W'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization specific to this dataset
    def _get_files_stats(self):
        super()._get_files_stats()
        with h5py.File(self.files_paths[0], 'r') as _f:
            self.img_shape_x_lr = _f['tasks']["u_lr"].shape[1]
            self.img_shape_y_lr = _f['tasks']["u_lr"].shape[2]
            _f.close()
        print(f"find {self.n_files} files at location {self.location}. Number of examples total: {self.n_samples_per_file}. To-use data per trajectory: {self.input_per_file} LR Image Shape: {self.img_shape_x_lr} x {self.img_shape_y_lr} x {self.n_in_channels}")

    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx)
        self.ensure_correct_index(global_idx, local_idx)

        if self.files[file_idx] is None:
            self._open_file(file_idx)

        channel_names = self.get_channel_names()
        lr_names = self.get_LR_channel_names()
        # Load the current sample
        y_current = self.load_sample(file_idx, local_idx,channel_names)
        X_start = self.load_sample(file_idx, local_idx,lr_names)
        # Load future samples
        y_future = self.load_future_samples(file_idx, local_idx,channel_names)
        X_end = self.load_future_samples(file_idx, local_idx,lr_names)[-1] # choose sampling conditon: this is last one 
        # Combine current and future samples
        y_samples = [y_current] + y_future
        X_samples = [X_start] + [X_end]
        y = torch.stack(y_samples, dim=1)
        X = torch.stack(X_samples, dim=1)
        return X, y

class GetDataset_diffIC_LowRes(BaseDataset):
    '''return B, T, C, H, W'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization specific to this dataset
    def _get_files_stats(self):
        super()._get_files_stats()
        with h5py.File(self.files_paths[0], 'r') as _f:
            self.img_shape_x_lr = _f['tasks']["u_lr"].shape[1]
            self.img_shape_y_lr = _f['tasks']["u_lr"].shape[2]
            _f.close()
        print(f"find {self.n_files} files at location {self.location}. Number of examples total: {self.n_samples_per_file}. To-use data per trajectory: {self.input_per_file} LR Image Shape: {self.img_shape_x_lr} x {self.img_shape_y_lr} x {self.n_in_channels}")

    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx)
        self.ensure_correct_index(global_idx, local_idx)

        if self.files[file_idx] is None:
            self._open_file(file_idx)

        channel_names = self.get_channel_names()
        lr_names = self.get_LR_channel_names()
        # Load the current sample
        y_current = self.load_sample(file_idx, local_idx,channel_names)
        X = self.load_sample(file_idx, local_idx,lr_names)
        # Load future samples
        y_future = self.load_future_samples(file_idx, local_idx,channel_names)
        # Combine current and future samples
        y_samples = [y_current] + y_future
        y = torch.stack(y_samples, dim=0)
        return X, y
    
class GetDataset_lres_viz(GetDataset_diffIC_LowRes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def _get_files_stats(self):
        super()._get_files_stats()

    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx)
        self.ensure_correct_index(global_idx, local_idx)
        if self.files[file_idx] is None:
            self._open_file(file_idx)

        channel_names = self.get_channel_names()
        lr_names = self.get_LR_channel_names()
        # Load the current sample
        y_current = self.load_sample(file_idx, local_idx,channel_names)
        X_current = self.load_sample(file_idx, local_idx,lr_names)
        # Load future samples
        X_future = self.load_future_samples(file_idx, local_idx,lr_names)
        y_future = self.load_future_samples(file_idx, local_idx,channel_names)
        # Combine current and future samples
        X_samples = [X_current] + X_future
        y_samples = [y_current] + y_future
        y = torch.stack(y_samples, dim=0)
        X = torch.stack(X_samples, dim=0)
        return X, y
    
class GetClimateDatasets(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.files_paths = glob.glob(self.location + "/data/*.h5")
        # Additional initialization specific to this dataset
    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/data/*.h5") #only take s9
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
        if self.files[file_idx] is None:
            self.files[file_idx] = h5py.File(self.files_paths[file_idx], 'r')['fields']
   
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
 
class GetClimateDatasets_special(GetClimateDatasets):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization specific to this dataset

    def __getitem__(self, global_idx):
        y_list = []
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

class GetDataset_diffIC_LowRes_crop(BaseDataset):
    '''return B, T, C, H, W'''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization specific to this dataset
    def _get_files_stats(self):
        super()._get_files_stats()
        with h5py.File(self.files_paths[0], 'r') as _f:
            self.img_shape_x_lr = _f['tasks']["u_lr"].shape[1]
            self.img_shape_y_lr = _f['tasks']["u_lr"].shape[2]
            _f.close()
        print(f"find {self.n_files} files at location {self.location}. Number of examples total: {self.n_samples_per_file}. To-use data per trajectory: {self.input_per_file} LR Image Shape: {self.img_shape_x_lr} x {self.img_shape_y_lr} x {self.n_in_channels}")
        n_crop_x = (self.img_shape_x // self.crop_size[0]) # Crop size of HR image
        n_crop_y = (self.img_shape_y // self.crop_size[1])
        self.patches = n_crop_x if n_crop_x == n_crop_y else 1
        print(f"number of patches: {self.patches}")
    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx)
        self.ensure_correct_index(global_idx, local_idx)

        if self.files[file_idx] is None:
            self._open_file(file_idx)

        channel_names = self.get_channel_names()
        lr_names = self.get_LR_channel_names()
        # Load the current sample
        y_current = self.load_sample(file_idx, local_idx,channel_names)
        X = self.load_sample(file_idx, local_idx,lr_names)
        # Load future samples
        y_future = self.load_future_samples(file_idx, local_idx,channel_names)
        # Combine current and future samples
        y_samples = [y_current] + y_future
        y = torch.stack(y_samples, dim=0)
        return X, y


class FNO_Special_Loader_Climate(GetClimateDatasets):
    """return B, C, T,H, W"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization specific to this dataset
        self.upsample = transforms.Resize((self.img_shape_x,self.img_shape_y),Image.BICUBIC,antialias=False)
    def __getitem__(self, global_idx):
        file_idx, local_idx = self.get_indices(global_idx)
        self.ensure_correct_index(global_idx, local_idx)
        gridx,gridy,gridt = torch.tensor(np.linspace(0,1,self.img_shape_x+1)[:-1]),torch.tensor(np.linspace(0,1,self.img_shape_y+1)[:-1]),torch.tensor(np.linspace(0,1,self.num_snapshots+1))
        gridx = gridx.reshape(1,1,self.img_shape_x,1).repeat([1,self.num_snapshots+1,1,self.img_shape_y])
        gridy = gridy.reshape(1,1,1,self.img_shape_y).repeat([1,self.num_snapshots+1,self.img_shape_x,1])
        gridt = gridt.reshape(1,self.num_snapshots+1,1,1).repeat([1,1,self.img_shape_x,self.img_shape_y])

        if self.files[file_idx] is None:
            self._open_file(file_idx)
        channel_names = self.get_channel_names()
        # Load the current sample
        y_current = self.load_sample(file_idx, local_idx,channel_names)
        X_lr = self.get_X(y_current)
        X_interp = self.upsample(X_lr)
        X_interp = X_interp.reshape(self.n_in_channels,1,self.img_shape_x,self.img_shape_y).repeat([1, self.num_snapshots+1, 1,1])
        X = torch.cat([X_interp,gridx,gridy,gridt],dim=0)
        # Load future samples
        y_future = self.load_future_samples(file_idx, local_idx,channel_names)
        # Combine current and future samples
        y_samples = [y_current] + y_future
        y = torch.stack(y_samples, dim=1)
        return X, y

   
if __name__ == "__main__":
    # for name in ["decay_turb","DT_FNO","DT_coord"]:
    #     train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(data_name= name,batch_size= 512,data_path="/pscratch/sd/j/junyi012/Decay_Turbulence_small",in_channels =3,timescale_factor= 20)
    #     for loader in [train_loader, val1_loader, val2_loader, test1_loader, test2_loader]:
    #         for idx, (input,target) in enumerate (loader):
    #             input = input
    #             target = target   
    #         print(f"{name} input shape {input.shape}")
    #         print(f"{name} target shape {target.shape}")
    for name in ["dcay_lrsim"]: #,"dy_sequenceLR"
        i =0 
        train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(data_name= name,batch_size= 512,data_path="../decay_turb_lrsim_short4",in_channels =1,timescale_factor= 10,num_snapshots=20)
        for loader in [train_loader, val1_loader, val2_loader, test1_loader, test2_loader]:
            for idx, (input,target) in enumerate (loader):
                input = input
                target = target   
            print(f"{name} input shape {input.shape}")
            print(f"{name} target shape {target.shape}")
            i +=1
            for j in range (5):
                fig,ax = plt.subplots(1,3)
                ax[0].imshow(input[-1,0,:,:])
                ax[1].imshow(target[-1,0,0,:,:])
                ax[2].imshow(target[-1,j*4,0,:,:])
                fig.savefig(f"debug/decay_turb{i}_{j}.png")
                plt.close()
    # for name in [ "climate","climate_sequence",]:
    #     train_loader, val1_loader, val2_loader, test1_loader, test2_loader  = getData(data_name= name,batch_size= 512,data_path="/pscratch/sd/j/junyi012/climate_data/s4_sig0/data",in_channels =3,timescale_factor= 20)
    #     for loader in [train_loader, val1_loader, val2_loader, test1_loader, test2_loader]:
    #         for idx, (input,target) in enumerate (loader):
    #             input = input
    #             target = target   
    #         print(f"{name} input shape {input.shape}")
    #         print(f"{name} target shape {target.shape}")