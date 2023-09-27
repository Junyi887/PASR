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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='nskt_16k', help='data name')
    parser.add_argument('--crop_size', type=int, default=128, help='cropsize')
    args = parser.parse_args()
    print(getNorm(args))
