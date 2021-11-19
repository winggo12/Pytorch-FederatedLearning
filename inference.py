import copy
import torch
import torch.nn as nn
from model.modelFcNet import FcNet
from model.modelFcNet import FcNet
import numpy as np
from datasetProcessing.dataset import TheDataset
from sklearn.metrics import confusion_matrix

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

if __name__ == '__main__':
    bank_dataset = TheDataset("./data/BankChurners_normalized_standardized.csv", train_ratio=0.9, train_or_val='test')
    data_size = bank_dataset.X_.shape[0]
    train_loader = torch.utils.data.DataLoader(bank_dataset, batch_size=data_size, shuffle=True)

    preds, ground_truth = inference_nn(model_path="./saved/model.pth", data_loader=train_loader)
    cm = confusion_matrix(ground_truth, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    print("------- result ------")
    print("Per-Class Acc: ")
    print(per_class_acc)
    print("Confusion Matrix: ")
    print(cm)