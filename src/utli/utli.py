import h5py
import numpy as np
import glob
import os
import argparse
# data path in raocp 
# DATA_DIR = {"rbc_diff_IC": ["../rbc_diff_IC/rbc_IC1/*.h5","../rbc_diff_IC/rbc_IC2/*.h5"],
#             "nskt_16k": ["../superbench/datasets/nskt16000_1024/train/*.h5", "../superbench/datasets/nskt16000_1024/valid_1/*.h5"],
#             "rbc_diff_10IC": ["../rbc_diff_IC/rbc_10IC/train/rbc_*_256/rbc_*_256_s9.h5","../rbc_diff_IC/rbc_10IC/test/rbc_*_256/rbc_*_256_s9.h5","../rbc_diff_IC/rbc_10IC/val/rbc_*_256/rbc_*_256_s9.h5"],
#             "climate":["../superbench/train/*.h5","../superbench/val/*.h5","../superbench/test/*.h5"],
#             "rbc_25664": ["../rbc_diff_IC/rbc_256_64/train/rbc_*_25664/rbc_*_25664_s2.h5","../rbc_diff_IC/rbc_256_64/val/rbc_*_25664/rbc_*_25664_s2.h5","../rbc_diff_IC/rbc_256_64/test/rbc_*_25664/rbc_*_25664_s2.h5"],
#             }
DATA_DIR = {"Decay_turb": ["../Decay_Turbulence/*/*.h5"],
            "rbc":["../rbc_10IC/*/*.h5"],
            "Burger2D":["../Burger2D/*/*.h5"],
            }
def CenterCrop(data, crop_size):
    h, w = data.shape[-2], data.shape[-1]
    h_crop, w_crop = (h-crop_size) // 2, (w-crop_size) // 2
    data = data[...,h_crop:h_crop+crop_size, w_crop:w_crop+crop_size]
    return data

def getNorm(args,method ="min-max"):
    # normalization should be consistant with the normalization in the training
    min = []
    max = []
    window_size = args.crop_size

    if method == "min-max":
        for data_path in DATA_DIR[args.data]:
            files_paths = glob.glob(data_path)
            files_paths.sort()
            data = []
            for fp in files_paths:
                with h5py.File(fp,'r') as _f:
                    if args.data == "nskt_16k":
                        HR_data = _f['fields'][()][:,2,:,:]
                        data.append(CenterCrop(HR_data, window_size))
                    else:
                        # TODO need to change to more channel, now only support 1 channel
                        if args.in_channel ==1:
                            HR_data = _f['tasks']['vorticity'][()]
                            data.append(CenterCrop(HR_data, window_size)) 
                        if args.in_channel ==2:
                            HR_data_u = _f['tasks']['u'][()]
                            HR_data_v = _f['tasks']['v'][()]

            data =np.concatenate(data, axis=0)
            data_min = np.min(data)
            data_max = np.max(data)
            min.append(data_min)
            max.append(data_max)
    
        data_min = np.min(np.stack(min,axis=0))
        data_max = np.max(np.stack(max,axis=0))
        print(f"min: {data_min}, max: {data_max}")
        return data_min,data_max
    elif method == "mean-std":
        for data_path in DATA_DIR[args.data]:
            files_paths = glob.glob(data_path)
            files_paths.sort()
            data = []
            for fp in files_paths:
                with h5py.File(fp,'r') as _f:
                    if args.data == "nskt_16k":
                        HR_data = _f['fields'][()][:,2,:,:]
                        data.append(CenterCrop(HR_data, window_size))
                    else:
                        # TODO need to change to more channel, now only support 1 channel
                        if args.in_channel ==1:
                            HR_data = _f['tasks']['vorticity'][()]
                            data.append(CenterCrop(HR_data, window_size)) 
                        if args.in_channel ==2:
                            HR_data_u = _f['tasks']['u'][()]
                            HR_data_v = _f['tasks']['v'][()]

            data =np.concatenate(data, axis=0)
            data_mean = np.mean(data)
            data_std = np.std(data)
            min.append(data_mean)
            max.append(data_std)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='nskt_16k', help='data name')
    parser.add_argument('--crop_size', type=int, default=128, help='cropsize')
    args = parser.parse_args()
    print(getNorm(args))
