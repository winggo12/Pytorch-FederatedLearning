import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def weighted_average_weights(w, dataset_proportions):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += torch.mul(w[i][key], dataset_proportions[i])
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg

def plot_partitions(user_num, label_partition_dict, alpha):
    user_partitions_dict = {}
    list = []
    for user_id in range(user_num):
        for label , partition in label_partition_dict.items():
            list.append(partition[user_id])
        user_partitions_dict[user_id] = list
        list = []

    plotdata = pd.DataFrame(label_partition_dict)
    plotdata.plot(kind='bar', stacked=True)
    title = "Alpha="+ str(alpha) + ",User = "+ str(user_num)
    plt.title(title)
    # plt.show()
    plt.savefig(title+".jpg")
    plt.close()
    return

def plot_acc_loss(acc_loss_dict, user_num, alpha):
    plotdata = pd.DataFrame(acc_loss_dict)
    acc_list, loss_list , epoches = [], [], []
    for epoch, dict in acc_loss_dict.items():
        epoches.append(epoch)
        acc_list.append(dict['acc'])
        loss_list.append(dict['loss'])
    plt.plot(epoches, acc_list)
    info = "Alpha="+ str(alpha) + ",User = "+ str(user_num)
    title = "Test Accuracy of Federated Learning "+info
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.savefig(title+".jpg")
    plt.close()

    plt.plot(epoches, loss_list)
    title = "Test Loss of Federated Learning "+info
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title(title)
    plt.savefig(title+".jpg")
    plt.close()


def train(alpha, user_num, global_rounds, local_epoches, batch_size, log = True):
    avg_global_test_acc , avg_global_test_loss = 0 , 0

    spliter = DatasetSplitByDirichletPartition(file_path=config.data_path,
                                               alpha=alpha,
                                               user_num=user_num,
                                               train_ratio=.7)
    dataset_dict = spliter.get_dataset_dict()
    label_partition_dict = spliter.get_label_partition_dict()
    dataset_size = spliter.get_complete_dataset_size()
    dataset_proportions = spliter.get_train_dataset_proportions()
    global_model = FcNet()
    plot_partitions(user_num=user_num, label_partition_dict=label_partition_dict,
                    alpha=alpha)
    global_acc_loss_dict = {}
    # global_acc_loss_dict[0] = {'acc': 0, 'loss': 0}
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
                                                local_epoch=local_epoches,
                                                batch_size = batch_size,
                                                log = log)
            local_weights.append(copy.deepcopy(model_weights))
            local_losses.append(loss)

        # global_weight = average_weights(local_weights)
        global_weight = weighted_average_weights(local_weights, dataset_proportions)
        global_model.load_state_dict(global_weight)



        test_acc, test_loss, test_per_class_acc = 0 , 0 , []

        for user_index in range(len(dataset_dict)):
            acc, loss , per_class_acc = inference(
                                            dataset=dataset_dict[user_index]['test'],
                                            model=global_model,
                                            batch_size=batch_size,
                                            log = log)
            if len(test_per_class_acc) == 0: test_per_class_acc = per_class_acc
            else : test_per_class_acc = [i+j for i,j in zip(test_per_class_acc,per_class_acc)]
            test_acc += acc
            test_loss += loss

        test_acc = test_acc/len(dataset_dict)
        test_loss = test_loss/len(dataset_dict)
        test_per_class_acc = [i/len(dataset_dict) for i in test_per_class_acc]
        global_acc_loss_dict[round_idx+1] = {'acc': test_acc, 'loss':test_loss}

        avg_global_test_acc += (test_acc*100)
        avg_global_test_loss += test_loss
        if log == True:
            print("_________________Global Trainer_________________________")
            print('Global Round :{}, the global accuracy is {:.3}%, and the global loss is {:.3}.'.format(round_idx, test_acc*100, test_loss))
            print('Per-class global acc is ', test_per_class_acc)
            print("________________________________________________________")

    avg_global_test_acc , avg_global_test_loss =  avg_global_test_acc/global_rounds, avg_global_test_loss/global_rounds

    print("_________________Final Result_________________________")
    print("Alpha = ", alpha, " User Number = ", user_num)
    print('Final global accuracy is {:.3}%, and the Final global loss is {:.3}.'.format(test_acc*100,test_loss))
    print('Average global accuracy is {:.3}%, and the Average global loss is {:.3}.'.format(avg_global_test_acc,avg_global_test_loss))
    print("________________________________________________________")
    # plot_partitions(user_num=user_num, label_partition_dict=label_partition_dict)
    plot_acc_loss(  acc_loss_dict=global_acc_loss_dict,
                    user_num=user_num,
                    alpha=alpha)

if __name__ == '__main__':
    # for i in range(10):
    #     print("Trial Number {:}".format(i))
    #     train(log=False)
    train(  alpha = 0.02,
            user_num = 5,
            global_rounds = 30,
            local_epoches = 5,
            batch_size = 64,
            log=True)