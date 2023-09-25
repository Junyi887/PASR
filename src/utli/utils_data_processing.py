"""
data_loading.py

(0) MinMaxScaler: Min Max normalizer
"""

## Necessary Packages
import numpy as np
import os
import torch
import controldiffeq


def to_tensor(data):
    return torch.from_numpy(data).float()


def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - datasets: original datasets

    Returns:
      - norm_data: normalized datasets
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data


# class TimeDataset_irregular(torch.utils.data.Dataset):
#     def __init__(self, seq_len, data_name, missing_rate=0.0):
#         base_loc = here / 'datasets'
#         loc = here / 'datasets' / (data_name + str(missing_rate))
#         if os.path.exists(loc):
#             tensors = load_data(loc)
#             self.train_coeffs = tensors['train_a'], tensors['train_b'], tensors['train_c'], tensors['train_d']
#             self.samples = tensors['data']
#             self.original_sample = tensors['original_data']
#             self.original_sample = np.array(self.original_sample)
#             self.samples = np.array(self.samples)
#             self.size = len(self.samples)
#         else:
#             if not os.path.exists(base_loc):
#                 os.mkdir(base_loc)
#             if not os.path.exists(loc):
#                 os.mkdir(loc)

#             if data_name == 'stock':
#                 data = np.loadtxt('./datasets/stock_data.csv', delimiter=",", skiprows=1)
#             elif data_name == 'energy':
#                 data = np.loadtxt('./datasets/energy_data.csv', delimiter=",", skiprows=1)


#             total_length = len(data)
#             data = data[::-1]
#             self.min_val = np.min(data, 0)
#             self.max_val = np.max(data, 0) - np.min(data, 0)

#             self.original_sample = []
#             norm_data = MinMaxScaler(data)
#             ori_seq_data = []

#             for i in range(len(norm_data) - seq_len + 1):
#                 x = norm_data[i: i + seq_len].copy()
#                 ori_seq_data.append(x)
#             idx = torch.randperm(len(ori_seq_data))
#             for i in range(len(ori_seq_data)):
#                 self.original_sample.append(ori_seq_data[idx[i]])
#             orig_samples_np = np.array(self.original_sample)
#             self.X_mean = np.mean(orig_samples_np, axis=0).reshape(1, orig_samples_np.shape[1], orig_samples_np.shape[2])

#             generator = torch.Generator().manual_seed(56789)
#             removed_points = torch.randperm(norm_data.shape[0], generator=generator)[
#                              :int(norm_data.shape[0] * missing_rate)].sort().values
#             norm_data[removed_points] = float('nan')
#             total_length = len(norm_data)
#             index = np.array(range(total_length)).reshape(-1, 1)
#             norm_data = np.concatenate((norm_data, index), axis=1)  # 맨 뒤에 관측시간에 대한 정보 저장
#             seq_data = []
#             for i in range(len(norm_data) - seq_len + 1):
#                 x = norm_data[i: i + seq_len]
#                 seq_data.append(x)
#             self.samples = []
#             for i in range(len(seq_data)):
#                 self.samples.append(seq_data[idx[i]])

#             self.samples = np.array(self.samples)

#             norm_data_tensor = torch.Tensor(self.samples[:, :, :-1]).float().cuda()

#             time = torch.FloatTensor(list(range(norm_data_tensor.size(1)))).cuda()
#             self.last = torch.Tensor(self.samples[:, :, -1][:, -1]).float()
#             self.train_coeffs = controldiffeq.natural_cubic_spline_coeffs(time, norm_data_tensor)
#             self.original_sample = torch.tensor(self.original_sample)
#             self.samples = torch.tensor(self.samples)

#             save_data(loc, data=self.samples,
#                       original_data=self.original_sample,
#                       train_a=self.train_coeffs[0],
#                       train_b=self.train_coeffs[1],
#                       train_c=self.train_coeffs[2],
#                       train_d=self.train_coeffs[3],
#                       )

#             self.original_sample = np.array(self.original_sample)
#             self.samples = np.array(self.samples)
#             self.size = len(self.samples)


#     def __getitem__(self, index):
#         # batch _idx -> batch 만큼 가져고
#         batch_coeff = (self.train_coeffs[0][index].float(),
#                        self.train_coeffs[1][index].float(),
#                        self.train_coeffs[2][index].float(),
#                        self.train_coeffs[3][index].float())

#         self.sample = {'data': self.samples[index], 'inter': batch_coeff, 'original_data': self.original_sample[index]}

#         return self.sample  # self.samples[index]

#     def __len__(self):
#         return len(self.samples)



# def data_loading(args):
#     # dataset parameters
#     import torch
#     import torch.utils.data as Data

#     args.seq_len = 24
#     if args.dataset in ['stock', 'energy']:
#         ori_data = real_data_loading(args.dataset, args.seq_len)
#     elif args.dataset == 'sine':
#         # Set number of samples and its dimensions
#         no, dim = 10000, 5
#         ori_data = sine_data_generation(no, args.seq_len, dim)
#     ori_data = torch.Tensor(np.array(ori_data))
#     args.inp_dim = ori_data.shape[-1]
#     generator = torch.Generator().manual_seed(42)
#     train_set, test_set = Data.random_split(ori_data, [0.7, 0.3], generator=generator)
#     train_loader = Data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
#     test_loader = Data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
#     return ori_data, test_loader, train_loader