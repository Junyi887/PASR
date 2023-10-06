import os
import h5py
import numpy as np
import glob

location = ""
files_paths = glob.glob(location + "../superbench/datasets/era5/test_2/2015.h5") #only take s
files_paths = sorted(files_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
n_files = len(files_paths)
print(files_paths)
print("Found {} files".format(n_files))

all_data = []
# Load data from each file and append to the list
for file_path in files_paths:
    with h5py.File(file_path, 'r') as _f:
        data = _f["fields"][()][:,:,0:400,900:1300]
        all_data.append(data)

# Concatenate all the data
combined_data = np.concatenate(all_data, axis=0)
print(combined_data.shape)
# Save the combined data to a new .h5 file
output_filename = "combined_climate_0715.h5"
with h5py.File(output_filename, 'w') as f_out:
    f_out.create_dataset("fields", data=combined_data)