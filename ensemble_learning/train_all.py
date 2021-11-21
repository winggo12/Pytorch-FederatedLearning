import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from decisiontree import dt_train
from knn import knn_train
from lda import lda_train
from xgb import xgbc_train
from train import train_nn
from dataset_processing.dataset import TheDatasetByDataframe

df_ts = pd.read_csv('../data/BankChurners_normalized_standardized.csv')

inputs = df_ts.iloc[0:,0:10]
labels = df_ts.iloc[0:,10:11]

X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=0.3,random_state=0)

X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)

#Training for Sklearn and Xgboost Model
dt_train(X_train, X_test, y_train, y_test)
knn_train(X_train, X_test, y_train, y_test)
lda_train(X_train, X_test, y_train, y_test)
xgbc_train(X_train, X_test, y_train, y_test)

#Training for Pytorch Model
bank_train_dataset = TheDatasetByDataframe(input_df=X_train, label_df=y_train)
bank_test_dataset = TheDatasetByDataframe(input_df=X_test, label_df=y_test)
train_loader = torch.utils.data.DataLoader(bank_train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(bank_test_dataset, batch_size=32, shuffle=True)
train_nn(model_save_path="model.pth", acc_save_path="model.txt",
         train_loader=train_loader, test_loader=test_loader)


