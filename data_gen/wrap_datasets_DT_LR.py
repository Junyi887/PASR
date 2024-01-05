import h5py
import numpy as np
import os
from scipy.stats import pearsonr
def get_correlation(w_lr,w_hr):
    correlations = np.zeros(w_hr.shape[0])
    for i in range(w_hr.shape[0]):
        correlations[i],_ = pearsonr(w_lr[i].flatten(),w_hr[i].flatten())
    return correlations
# Save the figure


HR_res = 1024
scale = 16
data_dir = "/pscratch/sd/j/junyi012/DT_multi_resolution_unpaired/"
saved_dir = f"/pscratch/sd/j/junyi012/DT_lrsim_{HR_res}_s{scale}_v0"
if not os.path.exists(saved_dir):
    os.makedirs(saved_dir)
for tag in ["/train","/test","/val"]: 
    if not os.path.exists(saved_dir+tag):
        os.makedirs(saved_dir+tag)
# load data 
tag_list = ["/train","/test","/val"]
# glob all the data
import glob
data_list = glob.glob(data_dir+ f"*_res_{HR_res}_*.h5")
data_list.sort()
data_list_lr = glob.glob(data_dir+ f"*_res_{HR_res//scale}_*.h5")
data_list_lr.sort() 
# training data
from tqdm import tqdm
for idx, (data_path,lr_path) in enumerate (tqdm(zip(data_list[2:-2],data_list_lr[2:-2]))):
    print(f"trimming data at {data_path}")
    if data_path.split("/")[-1].split("_")[-1] == lr_path.split("/")[-1].split("_")[-1]:
        data = h5py.File(data_path, 'r')
        data_lr = h5py.File(lr_path, 'r')
        u_lr = data_lr['tasks']['u'][250:]
        v_lr = data_lr['tasks']['v'][250:]
        vorticity_lr = data_lr['tasks']['vorticity'][250:]
        t = data['tasks']['t'][250:]
        u = data['tasks']['u'][250:]
        v = data['tasks']['v'][250:]
        vorticity = data['tasks']['vorticity'][250:]   
        # save new data 
        data.close()
        data_lr.close()
        saved_path = saved_dir+tag_list[0]+f'/DT_LR_{HR_res}_s{scale}_{data_path.split("/")[-1].split("_")[-1]}'
        print("saving data at ",saved_path)
        with h5py.File( saved_path,'w') as f:
            tasks = f.create_group('tasks')
            tasks.create_dataset('u_lr', data=u_lr)
            tasks.create_dataset('v_lr', data=v_lr)
            tasks.create_dataset('vorticity_lr', data=vorticity_lr)
            tasks.create_dataset('u', data=u)
            tasks.create_dataset('v', data=v)
            tasks.create_dataset('vorticity', data=vorticity)
            tasks.create_dataset('t', data=t)
            f.close()
        print(f"saved data at {saved_path}")
    else:
        raise ValueError("data not match")
    
for idx, (data_path,lr_path) in enumerate (tqdm(zip(data_list[:2],data_list_lr[:2]))):
    print(f"trimming data at {data_path}")
    if data_path.split("/")[-1].split("_")[-1] == lr_path.split("/")[-1].split("_")[-1]:
        data = h5py.File(data_path, 'r')
        data_lr = h5py.File(lr_path, 'r')
        u_lr = data_lr['tasks']['u'][250:]
        v_lr = data_lr['tasks']['v'][250:]
        vorticity_lr = data_lr['tasks']['vorticity'][250:]
        t = data['tasks']['t'][250:]
        u = data['tasks']['u'][250:]
        v = data['tasks']['v'][250:]
        vorticity = data['tasks']['vorticity'][250:]   
        # save new data 
        data.close()
        data_lr.close()
        saved_path = saved_dir+tag_list[1]+f'/DT_LR_{HR_res}_s{scale}_{data_path.split("/")[-1].split("_")[-1]}'
        print("saving data at ",saved_path)
        with h5py.File( saved_path,'w') as f:
            tasks = f.create_group('tasks')
            tasks.create_dataset('u_lr', data=u_lr)
            tasks.create_dataset('v_lr', data=v_lr)
            tasks.create_dataset('vorticity_lr', data=vorticity_lr)
            tasks.create_dataset('u', data=u)
            tasks.create_dataset('v', data=v)
            tasks.create_dataset('vorticity', data=vorticity)
            tasks.create_dataset('t', data=t)
            f.close()
        print(f"saved data at {saved_path}")
    else:
        raise ValueError("data not match")
    
for idx, (data_path,lr_path) in enumerate (tqdm(zip(data_list[-2:],data_list_lr[-2:]))):
    print(f"trimming data at {data_path}")
    if data_path.split("/")[-1].split("_")[-1] == lr_path.split("/")[-1].split("_")[-1]:
        data = h5py.File(data_path, 'r')
        data_lr = h5py.File(lr_path, 'r')
        u_lr = data_lr['tasks']['u'][250:]
        v_lr = data_lr['tasks']['v'][250:]
        vorticity_lr = data_lr['tasks']['vorticity'][250:]
        t = data['tasks']['t'][250:]
        u = data['tasks']['u'][250:]
        v = data['tasks']['v'][250:]
        vorticity = data['tasks']['vorticity'][250:]   
        # save new data 
        data.close()
        data_lr.close()
        saved_path = saved_dir+tag_list[-1]+f'/DT_LR_{HR_res}_s{scale}_{data_path.split("/")[-1].split("_")[-1]}'
        print("saving data at ",saved_path)
        with h5py.File( saved_path,'w') as f:
            tasks = f.create_group('tasks')
            tasks.create_dataset('u_lr', data=u_lr)
            tasks.create_dataset('v_lr', data=v_lr)
            tasks.create_dataset('vorticity_lr', data=vorticity_lr)
            tasks.create_dataset('u', data=u)
            tasks.create_dataset('v', data=v)
            tasks.create_dataset('vorticity', data=vorticity)
            tasks.create_dataset('t', data=t)
            f.close()
        print(f"saved data at {saved_path}")
    else:
        raise ValueError("data not match")
#    = data['tasks']['u'][250:]
#     data['tasks']['u_lr'][...] = data['tasks']['u_lr'][250:,:,:]
#     data['tasks']['v'][...] = data['tasks']['v'][250:]
#     data['tasks']['v_lr'][...] = data['tasks']['v_lr'][250:,:,:]
#     data['tasks']['vorticity'][...] = data['tasks']['vorticity'][250:]
#     data['tasks']['vorticity_lr'][...] = data['tasks']['vorticity_lr'][250:,:,:]
#     data['tasks']['t'][...] = data['tasks']['t'][250:]
#     # save new data 
#     data.close()
#     os.rename(data_path, saved_dir+tag[0]+'/'+data_path.split("/")[-1])
#     # visualize correlation 
# # testing data
# for data_path in data_list[:2]:
#     print(f"trimming data at {data_path}")
#     data = h5py.File(data_path, 'w')
#     data['tasks']['u'][...] = data['tasks']['u'][250:]
#     data['tasks']['u_lr'][...] = data['tasks']['u_lr'][250:,:,:]
#     data['tasks']['v'][...] = data['tasks']['v'][250:]
#     data['tasks']['v_lr'][...] = data['tasks']['v_lr'][250:,:,:]
#     data['tasks']['vorticity'][...] = data['tasks']['vorticity'][250:]
#     data['tasks']['vorticity_lr'][...] = data['tasks']['vorticity_lr'][250:,:,:]
#     data['tasks']['t'][...] = data['tasks']['t'][250:]
#     # save new data 
#     data.close()
#     os.rename(data_path, saved_dir+tag[1]+'/'+data_path.split("/")[-1])
#     # visualize correlation

# for data_path in data_list[2:4]:
#     print(f"trimming data at {data_path}")
#     data = h5py.File(data_path, 'w')
#     data['tasks']['u'][...] = data['tasks']['u'][250:]
#     data['tasks']['u_lr'][...] = data['tasks']['u_lr'][250:,:,:]
#     data['tasks']['v'][...] = data['tasks']['v'][250:]
#     data['tasks']['v_lr'][...] = data['tasks']['v_lr'][250:,:,:]
#     data['tasks']['vorticity'][...] = data['tasks']['vorticity'][250:]
#     data['tasks']['vorticity_lr'][...] = data['tasks']['vorticity_lr'][250:,:,:]
#     data['tasks']['t'][...] = data['tasks']['t'][250:]
#     # save new data 
#     data.close()
#     os.rename(data_path, saved_dir+tag[2]+'/'+data_path.split("/")[-1])
#     # visualize correlation
import matplotlib.pyplot as plt
import seaborn as sns     

# correlations = []
# scale =1
# for i in [1024,512,256,128,64,32]:
#     data_path = f"/pscratch/sd/j/junyi012/DT_multi_resolution_unpaired/DT_lres_sim_res_{i}_2405.h5"
#     data = h5py.File(data_path, 'r')
#     if i ==  1024:
#         w_hr = data['tasks']['vorticity'][:]
#         t = data['tasks']['t'][:]
#     w = data['tasks']['vorticity'][:]
#     cor = get_correlation(w_hr[:,::scale,::scale],w)
#     correlations.append(cor)
#     scale = scale *2
#     print(scale)
# j = 0
# baseline_palette = sns.color_palette('YlGnBu', n_colors=7)[1:]
# for correlation in correlations:
#     plt.plot(t,correlation,color=baseline_palette[-j])
#     j+=1

# plt.legend(["s=1024","s=512","s=256","s=128","s=64","s=32"])
# plt.axhline(0.8,linestyle="--",color="grey",alpha=0.2)
# plt.xlabel("Time")
# plt.ylabel("Correlation")
# plt.ylim(0,1.1)
# ax = plt.gca()
# t2 = np.arange(0,2000,1)
# def forward(x):
#     return x // (t[2] - t[1])
# def inverse(x):
#     return x * (t[2] - t[1])
# #Adding a secondary x-axis on top for displaying the index of 't'
# ax2 = ax.secondary_xaxis('top', functions=(forward, inverse))
# # Optionally, set labels and ticks for ax2
# ax2.set_xlabel("Iteration step")
# plt.savefig("correlation1.png")