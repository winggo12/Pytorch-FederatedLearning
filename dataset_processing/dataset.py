from __future__ import print_function
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

from config import config
from dataset_preprocessing.dataset_preprocess import dataset_preprocess

#Returning a numpy array of one-hot-key
def ConvertToOneHotKey(y):
    # For Classification : One Hot Key
    # num_of_class = y.max() - y.min() + 1
    num_of_class = config.num_class
    one_hot_key = np.zeros(num_of_class)

    y_one_hot_key = []
    for element in y:
        template = one_hot_key.copy().tolist()
        id = int(element[0] - 1)
        template[id] = 1
        y_one_hot_key.append(template)

    y_one_hot_key = np.array(y_one_hot_key)

    return y_one_hot_key

class TheDataset(Dataset):
    def __init__(self, file_path, train_or_val, train_ratio):
        if train_ratio<0 or train_ratio>1:
            raise NameError("Ratio is not in correct range")
        # df = pd.read_csv(file_path)
        df = dataset_preprocess(file_path)
        row_num, col_num = df.shape[1], df.shape[0]
        train_size = round(col_num * (train_ratio))
        self.train_size = train_size
        # Handling Input Data and GroundTruth
        if train_or_val == 'train':
            x = df.iloc[0:train_size, 0:row_num-1].values
            y = df.iloc[0:train_size, row_num - 1:row_num].values
        elif train_or_val == 'test':
            x = df.iloc[train_size+1:col_num, 0:row_num-1].values
            y = df.iloc[train_size+1:col_num, row_num - 1:row_num].values
        elif train_or_val == 'all':
            x = df.iloc[0:col_num, 0:row_num-1].values
            y = df.iloc[0:col_num, row_num - 1:row_num].values
        elif train_or_val == 'no_label':
            x = df.iloc[0:col_num, 0:row_num-1].values
            y = []
            for i in range(col_num): y.append([1])
            y = np.asarray(y)

        self.X_ = torch.tensor(x, dtype=torch.float32)

        self.Y_ = torch.tensor(ConvertToOneHotKey(y))

        #For Regression
        # self.Y_ = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y_)

    def __getitem__(self , idx):
        try:
            return self.X_[idx] , self.Y_[idx]
        except Exception as e:
            print("Error Reading ", e)

class TheDatasetByDataframe(Dataset):
    def __init__(self, input_df, label_df):
        x = input_df.values
        y = label_df.values

        self.X_ = torch.tensor(x, dtype=torch.float32)

        self.Y_ = torch.tensor(ConvertToOneHotKey(y))

        #For Regression
        # self.Y_ = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y_)

    def __getitem__(self , idx):
        try:
            return self.X_[idx] , self.Y_[idx]
        except Exception as e:
            print("Error Reading ", e)