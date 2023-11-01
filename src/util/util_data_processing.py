

## Necessary Packages
import numpy as np
import os
import torch
import glob
import h5py
def to_tensor(data):
    return torch.from_numpy(data).float()

import numpy as np
import h5py
import glob

class DataInfoLoader():
  """
  A class to load and process data statistical information from multiple datasets.

  DATA_DIR = {"Decay_turb": ["../Decay_Turbulence/*/*.h5"],
            "rbc":["../rbc_10IC/*/*.h5"],
            "Burger2D":["../Burger2D/*/*.h5"],
            }
  Attributes:
  - data_path: A string representing the path to the data.
  - mean_all_data: A numpy array representing the mean of all data.
  - std_all_data: A numpy array representing the standard deviation of all data.
  - max_all_data: A numpy array representing the maximum value of all data.
  - min_all_data: A numpy array representing the minimum value of all data.
  - n_files: An integer representing the number of files.
  - n_samples_per_file: An integer representing the number of samples per file.
  - img_shape_x: An integer representing the x shape of the image.
  - img_shape_y: An integer representing the y shape of the image.
  - files_paths: A list of strings representing the paths to the files.

  Methods:
  - __init__(self, data_path): Initializes the DataInfoLoader class.
  - get_all_data(): Returns all data.
  - get_min_max(self): Returns the minimum and maximum values of all data.
  - get_mean_std(self): Returns the mean and standard deviation of all data.
  - _get_files_stats(self): Gets the statistics of all files.
  """
  def __init__(self,data_path):
    """
    Initializes the DataInfoLoader class.

    Parameters:
    - data_path: A string representing the path to the data.
    """
    self.data_path = data_path
    self._get_files_stats()

  def get_all_data(self):
    """
    Returns all data.

    Returns:
    - All data.
    """
    return 0 

  def get_min_max(self):
    """
    Returns the minimum and maximum values of all data.

    Returns:
    - The minimum and maximum values of all data.
    """
    return self.min_all_data.min(axis = 0), self.max_all_data.max(axis = 0)

  def get_mean_std(self):
    """
    Returns the mean and standard deviation of all data.

    Returns:
    - The mean and standard deviation of all data.
    """
    counts = [1]*self.n_files
    mean_list = []
    std_list = []

    for i in range(3):
      means = self.mean_all_data[:,i]
      stds = self.std_all_data[:,i]
      total_count = sum(counts)
      combined_mean = np.sum([m * n for m, n in zip(means, counts)]) / total_count

      # Calculate the variance for the combined dataset
      combined_variance = np.sum([(counts[i] * stds[i]**2 + counts[i] * (means[i] - combined_mean)**2) for i in range(len(counts))]) / total_count
      mean_list.append(combined_mean)
      std_list.append(np.sqrt(combined_variance))
    return np.stack(mean_list), np.stack(std_list)

  def _get_files_stats(self):
    """
    Gets the statistics of all files.
    """
    self.mean_list = []
    self.std_list = []
    self.min_list = []
    self.max_list = []
    self.files_paths = glob.glob(self.data_path) #only take s9
    self.files_paths.sort()
    self.n_files = len(self.files_paths)
    print("Found {} files".format(self.n_files))
    for i in range(self.n_files):
      with h5py.File(self.files_paths[i], 'r') as _f:
        print("Getting file stats from {}".format(self.files_paths[i]))
        u = _f['tasks']["u"][()]
        v = _f['tasks']["v"][()]
        w = _f['tasks']["vorticity"][()]
        self.n_samples_per_file = _f['tasks']["u"].shape[0]
        self.img_shape_x = _f['tasks']["u"].shape[1]
        self.img_shape_y = _f['tasks']["u"].shape[2]
        _f.close()
      mean0 = w.mean()
      std0 = w.std()
      max0 = w.max()
      min0 = w.min()
      mean1 = u.mean()
      std1 = u.std()
      max1 = u.max()
      min1 = u.min()
      mean2 = v.mean()
      std2 = v.std()
      max2 = v.max()
      min2 = v.min()
      mean = np.stack((mean0,mean1,mean2)) #
      max = np.stack((max0,max1,max2))
      min = np.stack((min0,min1,min2)) 
      std = np.stack((std0,std1,std2))
      self.mean_list.append(mean)
      self.std_list.append(std)
      self.max_list.append(max)
      self.min_list.append(min)
    self.min_all_data = np.stack(self.min_list)
    self.max_all_data = np.stack(self.max_list)    
    self.mean_all_data = np.stack(self.mean_list)
    self.std_all_data = np.stack(self.std_list)


if __name__ == "__main__":
  print("hello")
  info = DataInfoLoader("../Decay_Turbulence_small/*/*.h5")
  print(info.get_mean_std())
  print(info.get_min_max())