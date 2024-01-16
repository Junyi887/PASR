import h5py
import numpy as np
import os
from scipy.stats import pearsonr
def get_correlation(w_lr,w_hr):
    correlations = np.zeros(w_hr.shape[0])
    for i in range(w_hr.shape[0]):
        correlations[i],_ = pearsonr(w_lr[i].flatten(),w_hr[i].flatten())
    return correlations
# Save the figur
sig = 0
data_dir = f"/pscratch/sd/j/junyi012/climate_data/s4_sig{sig}/data"
saved_dir = f"/pscratch/sd/j/junyi012/climate_sigma{sig}_v2"
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)
for tag in ["/train","/test","/val","/viz"]: 
    if not os.path.exists(saved_dir+tag):
        os.makedirs(saved_dir+tag)
# load data 
tag_list = ["/train","/test","/val","/viz"]
# glob all the data
import glob
data_list = glob.glob(data_dir+ f"/*.h5")
data_list.sort()
print(data_list)
# # training data
from tqdm import tqdm
for idx, data_path in enumerate (tqdm(data_list)):
    print(f"trimming data at {data_path}")
    data = h5py.File(data_path,'r')
    print(data.keys())
    print(data["fields"].shape)
    training_data = data["fields"][:365*8,:,:]
    print(training_data.shape)
    # save the data
    validation_data = data["fields"][365*8:365*8+365//2,:,:]
    test_data = data["fields"][365*8+365//2:,:,:]
    print(validation_data.shape)
    print(test_data.shape)
    # save the data
    f = h5py.File(saved_dir+tag_list[0]+"/climate_s4_train.h5", "w")
    f.create_dataset('fields', data=training_data)
    f.close()
    f = h5py.File(saved_dir+tag_list[1]+"/climate_s4_val.h5", "w")
    f.create_dataset('fields', data=validation_data)
    f.close()
    f = h5py.File(saved_dir+tag_list[2]+"/climate_s4_test.h5", "w")
    f.create_dataset('fields', data=test_data)
    f.close()
    f = h5py.File(saved_dir+"/viz"+"/climate_s4_viz.h5", "w")
    f.create_dataset('fields', data=data["fields"][365*8:,:,:])
    f.close()