import copy
import torch
import torch.nn as nn
from model.modelFcNet import FcNet
from model.modelFcNetRegression import FcNetRegression
import numpy as np
from ensemble_learning.sklearn_utils import save_acc_result_txt
import sklearn
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score

from config import config
from datasetProcessing.dataset import TheDataset

batch_size = 32
epoches = 30
iteration = 0

class Status:
    def __init__(self,name,dataloader):
        self.name = name
        self.dataloader = dataloader
        self.iteration = 0
        self.running_corrects = 0
        self.running_corrects_per_itr = 0
        self.running_loss = 0
        self.running_loss_per_itr = 0
        self.preds = np.asarray([])
        self.ground_truth = np.asarray([])

def train_nn(model_path, acc_path, train_loader, test_loader):
    model = FcNet()
    # For Regression
    # model = FcNetRegression()
    params_to_update = model.parameters()

    decayRate = 0.96
    loss_func = nn.CrossEntropyLoss()
    # For Regression
    # loss_func = nn.MSELoss()
    optimizer = torch.optim.AdamW(params_to_update, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                                  amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    trainstatus = Status('train', dataloader=train_loader)
    teststatus = Status('test', dataloader=test_loader)

    for epoch in range(epoches):
        for status in [trainstatus, teststatus]:

            if status.name == 'train':
                model.train()
            else:
                model.eval()

            for inputs , labels in status.dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

                if status.name == 'train':
                    loss.backward()
                    optimizer.step()

                #For Classification
                _, preds = torch.max(outputs, 1)
                _, ground_truth = torch.max(labels, 1)

                #For Regression
                # preds = torch.round(outputs)
                # ground_truth = labels
                np_preds, np_ground_truth = preds.numpy(), ground_truth.numpy()
                status.preds = np.append(status.preds, np_preds)
                status.ground_truth = np.append(status.ground_truth, np_ground_truth)

                # status.running_loss += loss.item()
                status.running_loss += loss.item() * inputs.size(0)
                status.running_loss_per_itr += loss.item() * inputs.size(0)
                status.running_corrects += torch.sum(preds.data == ground_truth.data)
                status.running_corrects_per_itr += torch.sum(preds.data == ground_truth.data)

                if status.iteration % 100 == 0 and status.iteration!=0:
                    acc = status.running_corrects_per_itr / (100 * batch_size)
                    loss = status.running_loss_per_itr / (100 * batch_size)
                    status.running_corrects_per_itr = 0
                    status.running_loss_per_itr = 0
                    if status.name == 'test':
                        print("----------------Test-----------------------")
                        print("Stage: ", status.name ," Iteration:", status.iteration, " Acc: ", acc, " Loss: ", loss)
                        print("-------------------------------------------")
                    else:
                        print("Stage: ", status.name, " Iteration:", status.iteration, " Acc: ", acc, " Loss: ", loss)
                status.iteration += 1

    #Result :
    for status in [trainstatus, teststatus]:
        acc = accuracy_score(status.ground_truth, status.preds)
        cm = confusion_matrix(status.ground_truth, status.preds)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        if(status.name == 'test'): save_acc_result_txt(filename=acc_path, acc_list=per_class_acc)
        print("-------", status.name, " result ------")
        print("Acc: ", acc)
        print("Per-Class Acc: ")
        print(per_class_acc)
        print("Confusion Matrix: ")
        print(cm)

    # Save Model:
    saved_model = copy.deepcopy(model.state_dict())
    torch.save(saved_model, model_path)

if __name__ == '__main__':
    bank_train_dataset = TheDataset(config.data_path, train_ratio=0.9, train_or_val='train')
    bank_test_dataset = TheDataset(config.data_path, train_ratio=0.9, train_or_val='test')
    train_loader = torch.utils.data.DataLoader(bank_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(bank_test_dataset, batch_size=batch_size, shuffle=True)
    train_nn(model_path="./saved/model.pth", acc_path="./saved/model.txt",
             train_loader=train_loader, test_loader=test_loader)