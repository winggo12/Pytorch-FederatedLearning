import copy
import torch
import torch.nn as nn
from model.modelFcNet import FcNet, DeeperFcNet
from model.modelFcNetRegression import FcNetRegression
import numpy as np
from ensemble_learning.sklearn_utils import save_acc_result_txt, display_result
import sklearn
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score
from config import config
from dataset_processing.dataset import TheDataset
import matplotlib.pyplot as plt

batch_size = 32
epoches = 10
iteration = 0

class Status:
    def __init__(self,name,dataloader):
        self.name = name
        self.dataloader = dataloader
        self.iteration = 0
        self.report_iterations = 0
        self.running_corrects = 0
        self.running_corrects_per_itr = 0
        self.running_loss = 0
        self.running_loss_per_itr = 0
        self.preds = np.asarray([])
        self.ground_truth = np.asarray([])
        self.acc_list, self.loss_list = [], []
        self.confusion_matrix = 0

def plot_acc_loss(model_name ,test_acc_list, train_acc_list, test_loss_list, train_loss_list):
    epoches = []
    for i in range(len(test_acc_list)):
        epoches.append(i)
    plt.plot(epoches, test_acc_list, "r-", label="test")
    plt.plot(epoches, train_acc_list, "g-", label="train")
    plt.legend()
    info = model_name
    title = "Accuracy of "+info
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.savefig(title+".jpg")
    plt.close()

    plt.plot(epoches, test_loss_list, "r-", label="test" )
    plt.plot(epoches, train_loss_list, "g-", label="train" )
    plt.legend()
    title = "Loss of "+info
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.title(title)
    plt.savefig(title+".jpg")
    plt.close()

def plot_cm(model_name, cm):
    for matrix in cm:
        fig = plt.figure()
        plt.matshow(cm)
        plt.title('Confusion Matrix of '+model_name)
        plt.colorbar()
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig('Confusion Matrix of ' +model_name+ '.jpg')

def train_nn(model_save_path, acc_save_path, train_loader, test_loader):
    # model = FcNet()
    model = DeeperFcNet()
    params_to_update = model.parameters()

    decayRate = 0.96
    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(params_to_update, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                                  amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    trainstatus = Status('train', dataloader=train_loader)
    teststatus = Status('test', dataloader=test_loader)
    trainstatus.report_iterations = int(int(len(train_loader.dataset)/batch_size)*0.99)
    teststatus.report_iterations = int(int(len(test_loader.dataset)/batch_size)*0.99)
    for epoch in range(epoches):
        for status in [trainstatus, teststatus]:
            status.preds, status.ground_truth = np.asarray([]), np.asarray([])
            if status.name == 'train':
                model.train()
            else:
                model.eval()
            num_of_data = 0
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
                num_of_data += inputs.size(0)
                status.running_loss += loss.item() * inputs.size(0)
                status.running_loss_per_itr += loss.item() * inputs.size(0)
                status.running_corrects += torch.sum(preds.data == ground_truth.data)
                status.running_corrects_per_itr += torch.sum(preds.data == ground_truth.data)

                if status.iteration == status.report_iterations and status.iteration!=0:
                    acc = status.running_corrects_per_itr / (status.report_iterations * batch_size)
                    loss = status.running_loss_per_itr / (status.report_iterations * batch_size)
                    status.acc_list.append(acc)
                    status.loss_list.append(loss)
                    status.running_corrects_per_itr = 0
                    status.running_loss_per_itr = 0
                    if status.name == 'test':
                        print("----------------Test-----------------------")
                        print("Stage: ", status.name ," Iteration:", status.iteration, " Acc: ", acc, " Loss: ", loss)
                        print("-------------------------------------------")
                    else:
                        print("Stage: ", status.name, " Iteration:", status.iteration, " Acc: ", acc, " Loss: ", loss)
                status.iteration += 1
            status.iteration = 0

    #Result :
    for status in [trainstatus, teststatus]:
        acc = accuracy_score(status.ground_truth, status.preds)
        cm = confusion_matrix(status.ground_truth, status.preds)
        status.confusion_matrix = cm
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        if(status.name == 'test'): save_acc_result_txt(filename=acc_save_path, acc_list=per_class_acc)
        print("-------", status.name, " result ------")
        print("Acc: ", acc)
        print("Per-Class Acc: ")
        print(per_class_acc)
        print("Confusion Matrix: ")
        print(cm)

    # Save Model:
    saved_model = copy.deepcopy(model.state_dict())
    torch.save(saved_model, model_save_path)

    return teststatus.acc_list, trainstatus.acc_list, teststatus.loss_list, trainstatus.loss_list, teststatus.confusion_matrix, trainstatus.confusion_matrix

if __name__ == '__main__':
    bank_train_dataset = TheDataset(config.data_path, train_ratio=0.7, train_or_val='train')
    bank_test_dataset = TheDataset(config.data_path, train_ratio=0.7, train_or_val='test')
    train_loader = torch.utils.data.DataLoader(bank_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(bank_test_dataset, batch_size=batch_size, shuffle=True)
    test_acc_list, train_acc_list, test_loss_list, train_loss_list, test_cm, train_cm=\
    train_nn(model_save_path="./saved/deeperfcnet.pth",
             acc_save_path="./saved/deeperfcnet.txt",
             train_loader=train_loader, test_loader=test_loader)
    plot_acc_loss(model_name="DeeperFcNet",
                  test_acc_list=test_acc_list,
                  train_acc_list=train_acc_list,
                  test_loss_list=test_loss_list,
                  train_loss_list=train_loss_list)
    plot_cm(model_name="DeeperFcNet",
            cm=test_cm)