import copy
import torch
import torch.nn as nn
from model.modelFcNet import FcNet
from model.modelFcNet import FcNet
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from dataset_processing.dataset import TheDataset, ConvertToOneHotKey

batch_size = 32
epoches = 2000
iteration = 0

running_loss = 0
running_corrects = 0
running_loss_per_itr = 0
running_corrects_per_itr = 0

def inference_nn(model_path, data_loader):
    model_path = model_path
    model = FcNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preds = np.asarray([])

    for inputs, labels in data_loader:
        outputs = model(inputs)

    _, preds = torch.max(outputs, 1)
    _, ground_truth = torch.max(labels, 1)

    np_preds, np_ground_truth = preds.numpy(), ground_truth.numpy()

    return np_preds, np_ground_truth

def inference_nn_by_df(model_path, input_df):
    model_path = model_path
    model = FcNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    preds = np.asarray([])
    inputs = torch.tensor(input_df.values, dtype=torch.float32)
    # labels = np.max(label_df.values, 1)

    output = model(inputs)
    _, preds = torch.max(output, 1)

    return preds

if __name__ == '__main__':
    file_path = "./data/New_BankChurners.csv"
    bank_dataset = TheDataset(file_path=file_path, train_ratio=0, train_or_val='no_label')
    data_size = bank_dataset.X_.shape[0]
    train_loader = torch.utils.data.DataLoader(bank_dataset, batch_size=data_size, shuffle=False)

    preds, ground_truth = inference_nn(model_path='./saved/model.pth', data_loader=train_loader)
    cm = confusion_matrix(ground_truth, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    df = pd.read_csv(file_path)
    df.drop(["CreditLevel"], axis=1, inplace=True)
    df['CreditLevel'] = preds.tolist()
    # for i in range(data_size):
    #     df = df.append({'CreditLevel': preds[i]}, ignore_index=True)

    df.to_csv('./data/your_student_ID.csv', index=False)


