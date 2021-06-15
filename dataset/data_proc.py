import torch.utils.data as Data
import torch
import numpy as np
import random


class HAR_1d(Data.Dataset):

    def __init__(self, filename_x, filename_y):
        self.filename_x = filename_x
        self.filename_y = filename_y

    def __len__(self):
        return self.length

    def HAR_data_1d(self):
        data_x_raw = np.load(self.filename_x)
        data_x = data_x_raw.transpose(0, 2, 1)  # 7352 9 128 # (N, C, W)
        data_y = np.load(self.filename_y)
        torch_dataset = Data.TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
        return torch_dataset


class HAR(Data.Dataset):

    def __init__(self, filename_x, filename_y):
        self.filename_x = filename_x
        self.filename_y = filename_y

    def __len__(self):
        return self.length

    def HAR_data(self):
        # data_x_raw = np.load(self.filename_x)[1]  #####
        data_x_raw = np.load(self.filename_x)
        # data_x = data_x_raw.reshape(-1, 1, data_x_raw.shape[1], data_x_raw.shape[2])  # (N, C, H, W) (1964, 1, 128, 9)
        data_x = data_x_raw.reshape(-1, 1, 128, 9)  # (N, C, H, W) (7352, 1, 128, 9)
        # print(data_x.shape)
        # data_y = np.load(self.filename_y)[1].reshape(-1)  #####
        data_y = np.load(self.filename_y)
        torch_dataset = Data.TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
        return torch_dataset


class HAR_one_tensor(Data.Dataset):

    def __init__(self, filename_x, filename_y):
        self.filename_x = filename_x
        self.filename_y = filename_y

    def __len__(self):
        return self.length

    def HAR_one_tensor_data(self):
        i = random.randint(0, 1963)
        data_x_raw = np.load(self.filename_x)[i]  #####  (1964, 1, 128, 9)
        data_x = data_x_raw.reshape(-1, 1, 128, 9)  # (N, C, H, W) (7352, 1, 128, 9)
        print(i)
        data_y = np.load(self.filename_y)[i].reshape(-1)  #####
        print("True result", data_y)
        torch_dataset = Data.TensorDataset(torch.from_numpy(data_x), torch.from_numpy(data_y))
        return torch_dataset


if __name__ == "__main__":
    # train_x_list = '../dataset/UCI/x_train.npy'
    # train_y_list = '../dataset/UCI/y_train.npy'
    train_x_list = '../dataset/pamamp2/x_train.npy'
    train_y_list = '../dataset/pamamp2/y_train.npy'
    data_train = HAR(train_x_list, train_y_list)
    har_train_tensor = data_train.HAR_data()

