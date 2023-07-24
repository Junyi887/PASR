import h5py
import numpy as np
import torch
from scipy.interpolate import CubicSpline
import torch.nn.functional as F
import matplotlib.pyplot as plt

def first_t_then_x(file_path, crop, space_scale, time_scale):
    # Window calculation
    window = (1024 - crop) // 2
    window_end = window + crop

    # Loading file and processing data
    with h5py.File(file_path, 'r') as file:
        train_hr = file['fields'][()][:, -1, window:window_end, window:window_end]

    x_lr_t_lr = train_hr[::time_scale, ::space_scale, ::space_scale]
    x_lr_t_hr = train_hr[:, ::space_scale, ::space_scale]

    # Preparing interpolation
    t_hr = np.linspace(0, 1, train_hr.shape[0])
    t_lr = np.linspace(0, 1, x_lr_t_lr.shape[0])
    cs = CubicSpline(t_lr, x_lr_t_lr)

    # Interpolation and reshaping
    train_poly_pred = torch.tensor(cs(t_hr)).unsqueeze(1)
    train_poly_truth = torch.tensor(train_hr).unsqueeze(1)
    pred = F.interpolate(train_poly_pred, size=(crop, crop), mode='bicubic', align_corners=False)

    # Error calculation
    RFNE = torch.stack([
        torch.norm(pred[i] - train_poly_truth[i]) / torch.norm(train_poly_truth[i])
        for i in range(len(pred))
        if i % time_scale != 0
    ])

    pred = pred[200:280, 0, 100, 100].numpy()
    train_poly_truth = train_poly_truth[200:280, 0, 100, 100].numpy()

    plt.figure()
    plt.plot(pred, marker="o", label="pred")
    plt.plot(train_poly_truth, marker="o", label="truth")
    plt.legend()
    plt.savefig("test2.png")

    plt.figure()
    plt.plot(
        train_poly_pred[200:280, 0, 25, 25].squeeze().numpy(),
        marker="o",
        label="pred"
    )
    plt.plot(x_lr_t_hr[200:280, 25, 25], marker="o", label="truth")
    plt.legend()
    plt.savefig("test3.png")

    RFNE2 = torch.stack([
        torch.norm(
            torch.tensor(train_poly_pred[i].squeeze()) - torch.tensor(x_lr_t_hr[i])
        ) / torch.norm(torch.tensor(x_lr_t_hr[i].squeeze()))
        for i in range(x_lr_t_hr.shape[0])
        if i % time_scale != 0
    ])

    return torch.mean(RFNE), torch.mean(RFNE2)


def first_x_then_t(file_path, crop, space_scale, time_scale):
    # Window calculation
    window = (1024 - crop) // 2
    window_end = window + crop

    # Loading file and processing data
    with h5py.File(file_path, 'r') as file:
        train_hr = file['fields'][()][:, -1, window:window_end, window:window_end]

    x_lr_t_lr = train_hr[::time_scale, ::space_scale, ::space_scale]

    # Preparing interpolation
    x_lr_t_lr = torch.tensor(x_lr_t_lr).unsqueeze(1)
    pred_x = F.interpolate(x_lr_t_lr, size=(crop, crop), mode='bicubic', align_corners=False).squeeze()
    t_hr = np.linspace(0, 1, train_hr.shape[0])
    t_lr = np.linspace(0, 1, x_lr_t_lr.shape[0])
    cs = CubicSpline(t_lr, pred_x.numpy())
    pred = torch.tensor(cs(t_hr)).unsqueeze(1)
    train_poly_truth = torch.tensor(train_hr).unsqueeze(1)

    # Error calculation
    RFNE = torch.stack([
        torch.norm(pred[i] - train_poly_truth[i]) / torch.norm(train_poly_truth[i])
        for i in range(len(pred))
        if i % time_scale != 0
    ])

    return torch.mean(RFNE)




def data_RFNE()
# Use the function
file_path = '../superbench/datasets/nskt16000_1024/train/nskt_train.h5'
crop = 256
space_scale = 4
time_scale = 4
# mean_RFNE, mean_RFNE2 = first_t_then_x(file_path, crop, space_scale, time_scale)
# mean_RFNE2 = first_x_then_t(file_path, crop, space_scale, time_scale)
print(mean_RFNE)
print(mean_RFNE2)
