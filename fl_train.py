import copy
import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

from config import config
from datasetProcessing.fl_dataset import DatasetSplitByDirichletPartition
from model.modelFcNet import FcNet
from model.modelFcNetRegression import FcNetRegression
from trainer.trainer import local_trainer, inference

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg

def weighted_average_weights(w, partitions):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += torch.mul(w[i][key], partitions[i])
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg

def train(log = True):
    alpha = 0.1
    user_num = 5
    global_rounds = 10
    local_epochs = 5
    batch_size = 4
    avg_global_test_acc , avg_global_test_loss = 0 , 0

    spliter = DatasetSplitByDirichletPartition(file_path=config.data_path,
                                               alpha=alpha,
                                               user_num=user_num,
                                               train_ratio=.8)
    dataset_dict = spliter.get_dataset_dict()
    partitions = spliter.get_partitions()
    dataset_size = spliter.get_complete_dataset_size()

    global_model = FcNet()

    for round_idx in range(global_rounds):
        local_weights = []
        local_losses = []
        global_acc = []

        for user_index in range(user_num):
            local_dataset_size = dataset_dict[user_index]['train'].X_.shape[0]
            if log == True:
                print("_________________Local Trainer w/ size: ", local_dataset_size, "/",
                      dataset_size, "_________________________")
            model_weights, loss = local_trainer(dataset=dataset_dict[user_index]['train'],
                                                model=copy.deepcopy(global_model),
                                                global_round=round_idx,
                                                local_epoch=local_epochs,
                                                batch_size = batch_size,
                                                log = log)
            local_weights.append(copy.deepcopy(model_weights))
            local_losses.append(loss)

        # global_weight = average_weights(local_weights)
        global_weight = weighted_average_weights(local_weights, partitions)
        global_model.load_state_dict(global_weight)

        test_acc, test_loss = inference(dataset=dataset_dict[user_index]['test'],
                                        model=global_model,
                                        batch_size=batch_size,
                                        log = log)

        avg_global_test_acc += (test_acc*100)
        avg_global_test_loss += test_loss
        if log == True:
            print("_________________Global Trainer_________________________")
            print('Global Round :{}, the global accuracy is {:.3}%, and the global loss is {:.3}.'.format(round_idx, 100 * test_acc, test_loss))
            print("________________________________________________________")

    avg_global_test_acc , avg_global_test_loss =  avg_global_test_acc/global_rounds, avg_global_test_loss/global_rounds

    print("_________________Final Result_________________________")
    print("Partitions = ", partitions)
    print("Alpha = ", alpha, " User Number = ", user_num)
    print('Final global accuracy is {:.3}%, and the Final global loss is {:.3}.'.format(test_acc*100,test_loss))
    print('Average global accuracy is {:.3}%, and the Average global loss is {:.3}.'.format(avg_global_test_acc,avg_global_test_loss))
    print("________________________________________________________")

if __name__ == '__main__':
    for i in range(10):
        print("Trial Number {:}".format(i))
        train(log=False)