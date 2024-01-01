

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
import os
import json
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
  def __init__(self, data_path, data_name="Decay_turb"):
      self.data_path = data_path
      self.data_name = data_name

      npy_file_path = f"src/util/data_stats_{self.data_name}.npy"
      # if file exists and is not empty
      if os.path.exists(npy_file_path) and os.path.getsize(npy_file_path) > 0:
          self._load_stats_from_json(npy_file_path)
      else:
          self._get_files_stats()
          self._save_stats_to_npy()

  def _save_stats_to_npy(self):
    """
    Saves the calculated statistics to a JSON file.
    """
    stats = {
        "mean": self.mean_all_data,
        "std": self.std_all_data,
        "min": self.min_all_data,
        "max": self.max_all_data,
        "img_shape_x": self.img_shape_x,
        "img_shape_y": self.img_shape_y,
        "n_files": self.n_files,
        "n_samples_per_file": self.n_samples_per_file
    }
    np.save(f"src/util/data_stats_{self.data_name}.npy", stats)
    

  def _load_stats_from_npy(self, file_path):
    """
    Loads the statistics from a .npy file.

    Parameters:
    - file_path: Path to the .npy file containing the statistics.
    """
    stats =np.load(f"src/util/data_stats_{self.data_name}.npy", allow_pickle=True).item()
    self.mean_list = stats["mean"]
    self.std_list = stats["std"]
    self.min_list = stats["min"]
    self.max_list = stats["max"]
    self.img_shape_x = stats["img_shape_x"]
    self.img_shape_y = stats["img_shape_y"]
    self.n_files = stats["n_files"]
    self.n_samples_per_file = stats["n_samples_per_file"]

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
    if "climate" in self.data_name:
      return np.array([self.min_all_data.min(axis = 0)]), np.array([self.max_all_data.max(axis = 0)])
    else:
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
    if "climate" in self.data_name:
      mean_list.append(self.mean_all_data.mean(axis = 0))
      std_list.append(self.std_all_data.mean(axis = 0))
      return np.stack(mean_list), np.stack(std_list)
    else:
      for i in range(self.mean_all_data.shape[1]):
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
    if self.data_name.startswith("climate"):
      print("Found {} files".format(self.n_files))
      for i in range(self.n_files):
        with h5py.File(self.files_paths[i], 'r') as _f:
          print("Getting file stats from {}".format(self.files_paths[i]))
          w = _f['fields'][()]
          self.n_samples_per_file = _f['fields'].shape[0]
          self.img_shape_x = _f['fields'].shape[1]
          self.img_shape_y = _f['fields'].shape[2]
          _f.close()
        mean0 = w.mean()
        std0 = w.std()
        max0 = w.max()
        min0 = w.min()
        self.mean_list.append(mean0)
        self.std_list.append(std0)
        self.max_list.append(max0)
        self.min_list.append(min0)
      self.min_all_data = np.stack(self.min_list)
      self.max_all_data = np.stack(self.max_list)    
      self.mean_all_data = np.stack(self.mean_list)
      self.std_all_data = np.stack(self.std_list)
      print(f"mean shape {self.mean_all_data.shape}")
    else: 
      print("Found {} files".format(self.n_files))
      for i in range(self.n_files):
        with h5py.File(self.files_paths[i], 'r') as _f:
          print("Getting file stats from {}".format(self.files_paths[i]))
          u = _f['tasks']["u"][()].astype(np.float32)
          v = _f['tasks']["v"][()].astype(np.float32)
          w = _f['tasks']["vorticity"][()].astype(np.float32)
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

  def get_shape(self):
    return self.img_shape_x, self.img_shape_y

if __name__ == "__main__":
  print("hello")
  info = DataInfoLoader("/pscratch/sd/j/junyi012/climate_data/s4_sig4/data/*.h5",data_name= "climate")
  mean, std = info.get_mean_std()
  print()
  print(info.get_mean_std())
  print(info.get_min_max())
  mean,std = mean[0:1].tolist(),std[0:1].tolist()
  print(type(mean))
  print(len(mean))