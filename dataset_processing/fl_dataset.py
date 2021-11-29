from __future__ import print_function
import pandas as pd
import math
import numpy as np
from sklearn.utils import shuffle
import sys
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
    def __init__(self, in_tensor, out_tensor):
        self.X_ = in_tensor
        self.Y_ = out_tensor

    def get_dataset_size(self):
        return self.X_.shape[0]

    def __len__(self):
        return len(self.Y_)
    def __getitem__(self , idx):
        try:
            return self.X_[idx] , self.Y_[idx]
        except Exception as e:
            print("Error Reading ", e)

class DatasetSplitByDirichletPartition():
    def __init__(self, file_path, alpha, user_num, train_ratio):
        if train_ratio<=0 or train_ratio>1:
            raise NameError("Ratio is not in correct range")
        # df = pd.read_csv(file_path)
        df = dataset_preprocess(file_path)
        df_new = df.sort_values(by=['CreditLevel'])
        row_num, col_num = df.shape[1], df.shape[0]
        self.__dataset_size = col_num
        #Get the number of labels and create a dictionary
        #that stores such info
        #e.g. {0: 100, 1:200}
        labels = list(df_new["CreditLevel"].unique())
        label_num_dict = {}
        for label in labels:
            count = df_new[df_new["CreditLevel"] == label].shape[0]
            label_num_dict[label] = count

        label_id_dict = {}
        label_sizes = []
        count = 0

        #Get the id of all data and store them
        #by its label in a dictionary
        #e.g. {0: [1,2,3], 1:[4,5,6]}
        for k, v in label_num_dict.items():
            id_list = []
            for i in range(v):
                id_list.append(count)
                count += 1
            label_id_dict[k] = id_list
            label_sizes.append(len(id_list))

        #Find the minimum size of data of a
        #certain label
        min_size = min(label_sizes)
        min_partition = 1 / min_size

        #Using dirichlet distribution to distribute
        #the data to users and put them into dict
        #label_id_dict_of_users : eg {0: {0:[1],1:[4]},1: {0: [2,3],1:[5,6]}}

        label_partition_dict = {}
        for user in range(user_num):
            for label_index in range(len(labels)):
                label = label_index+1
                label_partition_dict[label] = np.random.dirichlet(np.repeat(alpha, user_num))

        self.__label_partition_dict = label_partition_dict
        label_id_dict_of_users = {}
        label_id_dict_of_user = {}
        # user_id = 0
        for user_id in range(user_num):
            user_data_size = 0
            for label, num in label_num_dict.items():
                number = num * label_partition_dict[label][user_id]
                user_data_size += int(number)
                id_list = [label_id_dict[label].pop(0) for idx in range(int(number))]
                label_id_dict_of_user[label] = id_list
            if user_data_size <= 0:
                print("A User Cannot get any data due to a low alpha value, please run again...")
                sys.exit("Error of Dirichlet Distribution")
            # List of label_id_dict for the users
            label_id_dict_of_users[user_id] = label_id_dict_of_user
            label_id_dict_of_user = {}
            # user_id += 1

        #Get the data with the index id of different classes
        dfs = []
        for user_id in range(len(label_id_dict_of_users)):
            ids = []
            for label, id in label_id_dict_of_users[user_id].items():
                ids = ids + id
            df_one = df_new.filter(items=ids, axis=0)
            dfs.append(df_one)

        #Create train and validation dataloader dict
        #for every user
        #e.g. { 0: {'train':DL , 'test':DL } ... }
        dataloader_dict = {}
        user_id_df_dict = {}
        user_id = 0
        for df in dfs:
            df = shuffle(df)
            row_num, col_num = df.shape[1], df.shape[0]
            train_size = round(col_num * (train_ratio))
            train_x = df.iloc[0:train_size, 0:row_num - 1].values
            train_y = df.iloc[0:train_size, row_num - 1:row_num].values
            val_x = df.iloc[train_size + 1:col_num, 0:row_num - 1].values
            val_y = df.iloc[train_size + 1:col_num, row_num - 1:row_num].values
            df_dict = {'train': {'x':pd.DataFrame(train_x),'y':pd.DataFrame(train_y)},
                       'test':  {'x':pd.DataFrame(train_x),'y':pd.DataFrame(train_y)}}
            train_x = torch.tensor(train_x, dtype=torch.float32)
            train_y = torch.tensor(ConvertToOneHotKey(train_y))
            val_x = torch.tensor(val_x, dtype=torch.float32)
            val_y = torch.tensor(ConvertToOneHotKey(val_y))
            train_val_dataloader = {'train':TheDataset(train_x,train_y),
                                    'test':TheDataset(val_x,val_y)}
            dataloader_dict[user_id] = train_val_dataloader
            user_id_df_dict[user_id] = df_dict
            user_id += 1

        self.dataloader_dict = dataloader_dict
        self.user_id_df_dict = user_id_df_dict

    def get_train_dataset_proportions(self):
        dataset_proportions = []
        train_dataset_total_num = 0
        for i in range(len(self.dataloader_dict)):
            train_dataset_total_num += len(self.dataloader_dict[i]['train'])

        for i in range(len(self.dataloader_dict)):
            proportion = len(self.dataloader_dict[i]['train'])/train_dataset_total_num
            dataset_proportions.append(proportion)

        return dataset_proportions

    def get_dataset_dict(self):
        return self.dataloader_dict

    def get_label_partition_dict(self):
        return self.__label_partition_dict

    def get_complete_dataset_size(self):
        return self.__dataset_size

    def save_users_df_as_csv(self, path):
        for user_id, df_dict in self.user_id_df_dict.items():
            X_train, y_train = df_dict['train']['x'], df_dict['train']['y']
            X_test, y_test = df_dict['test']['x'], df_dict['test']['y']
            csv_name = "/user_"+str(user_id)+"_"
            X_train.to_csv(path+csv_name+'X_train.csv', index=False)
            X_test.to_csv(path+csv_name+'X_test.csv', index=False)
            y_train.to_csv(path+csv_name+'y_train.csv', index=False)
            y_test.to_csv(path+csv_name+'y_test.csv', index=False)

if __name__ == '__main__':
    user_num = 5
    alpha = 0.02
    file_path = '../data/BankChurners_normalized_standardized.csv'
    spliter = DatasetSplitByDirichletPartition(file_path=file_path,
                                               alpha=alpha,
                                               user_num=user_num,
                                               train_ratio=.7)
    dataset_dict = spliter.get_dataset_dict()
    proportions = spliter.get_train_dataset_proportions()
    label_partition_dict = spliter.get_label_partition_dict()

    spliter.save_users_df_as_csv(path="../data/non_iid_data")
    print("End")

