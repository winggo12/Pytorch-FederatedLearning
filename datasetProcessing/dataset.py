from __future__ import print_function
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TheDataset(Dataset):
    def __init__(self, file_path, train_or_test, split_ratio):
        if split_ratio<=0 or split_ratio>1:
            raise NameError("Ratio is not in correct range")
        df = pd.read_csv(file_path)
        print(df.head())
        row_num, col_num = df.shape[1], df.shape[0]
        train_size = round( col_num*(split_ratio) )
        # Handling Input Data and GroundTruth
        if train_or_test == 'train' or split_ratio >= 1:
            x = df.iloc[0:train_size, 0:row_num-1].values
            y = df.iloc[0:train_size, row_num - 1:row_num].values
        else:
            x = df.iloc[train_size+1:col_num, 0:row_num-1].values
            y = df.iloc[train_size+1:col_num, row_num - 1:row_num].values

        self.X_ = torch.tensor(x, dtype=torch.float32)

        #For Classification : One Hot Key
        num_of_class = y.max() - y.min() + 1
        one_hot_key = np.zeros(num_of_class)

        y_one_hot_key = []
        for element in y:
            template = one_hot_key.copy().tolist()
            id = element[0]-1
            template[id] = 1
            y_one_hot_key.append(template)

        y_one_hot_key = np.array(y_one_hot_key)
        self.Y_ = torch.tensor(y_one_hot_key)

        #For Regression
        self.Y_ = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y_)

    def __getitem__(self , idx):
        try:
            return self.X_[idx] , self.Y_[idx]
        except Exception as e:
            print("Error Reading ", e)

