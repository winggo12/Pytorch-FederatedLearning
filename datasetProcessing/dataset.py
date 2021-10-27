from __future__ import print_function
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TheDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path)
        # df = df[df['Geography'] == 'France']
        print(df.head())
        row_num, col_num = df.shape[1], df.shape[0]
        train_size = round( col_num*(0.8) )

        #Handling Input Data
        x = df.iloc[0:train_size, 0:row_num-1].values
        self.X_train = torch.tensor(x, dtype=torch.float32)

        #Handling Output / GroundTruth
        y = df.iloc[0:train_size, row_num-1:row_num].values

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
        self.Y_train = torch.tensor(y_one_hot_key)

        #For Regression
        # self.Y_train = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.Y_train)

    def __getitem__(self , idx):
        try:
            return self.X_train[idx] , self.Y_train[idx]
        except Exception as e:
            print("Error Reading ", e)

