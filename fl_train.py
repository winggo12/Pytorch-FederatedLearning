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
from dataset_processing.fl_dataset import DatasetSplitByDirichletPartition
from model.modelFcNet import FcNet
from model.modelFcNetRegression import FcNetRegression
from trainer.trainer import local_trainer, inference

def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg

def weighted_average_weights(w, dataset_proportions):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(0, len(w)):
            w_avg[key] += torch.mul(w[i][key], dataset_proportions[i])
        w_avg[key] = torch.div(w_avg[key].float(), len(w))
    return w_avg

def ratio_of_sign(w_global, w_local):
    w_global_copy , w_local_copy = \
        copy.deepcopy(w_global) , copy.deepcopy(w_local)
    sign , sign_total= {} , 0
    for key in w_global_copy.keys():
        delta = torch.add(w_local_copy[key], -w_global_copy[key])
        delta = torch.flatten(delta)
        sign[key] = 0
        for element in delta.numpy():
            if element > 0 :
                    sign[key] += 1
                    sign_total += 1
            if element < 0 :
                    sign[key] -= 1
                    sign_total -= 1
    sign['total'] = sign_total

    return sign

def plot_partitions(user_num, label_partition_dict, alpha, path="", rlr=False):
    user_partitions_dict = {}
    list = []
    for user_id in range(user_num):
        for label , partition in label_partition_dict.items():
            list.append(partition[user_id])
        user_partitions_dict[user_id] = list
        list = []

    plotdata = pd.DataFrame(label_partition_dict)
    plotdata.plot(kind='bar', stacked=True)
    rlr_info = ""
    if rlr == True : rlr_info = " using RLR "
    title = "Alpha="+ str(alpha) + ",User = "+ str(user_num)+rlr_info
    plt.title(title)
    # plt.show()
    plt.savefig(path + title+".jpg")
    plt.close()
    return

def plot_global_acc_loss(global_acc_loss_dict, user_num, alpha, rlr=False):
    plotdata = pd.DataFrame(global_acc_loss_dict)
    acc_list, loss_list , epoches = [], [], []
    for epoch, dict in global_acc_loss_dict.items():
        epoches.append(epoch)
        acc_list.append(dict['acc'])
        loss_list.append(dict['loss'])
    plt.plot(epoches, acc_list)
    rlr_info = ""
    if rlr == True : rlr_info = " using RLR "
    info = "Alpha="+ str(alpha) + ",User = "+ str(user_num)
    title = "Test Accuracy of Federated Learning "+ rlr_info +info
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.savefig(title+".jpg")
    plt.close()

    plt.plot(epoches, loss_list)
    title = "Test Loss of Federated Learning "+ rlr_info +info
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.title(title)
    plt.savefig(title+".jpg")
    plt.close()

def plot_local_acc_loss(local_acc_loss_dict, user_num, alpha, rlr=False):
    # plt_acc, plt_loss = plt.figure(1), plt.figure(2)
    for item in ["acc", "loss"]:
        for user, acc_loss_dict in local_acc_loss_dict.items():
            for key, list in acc_loss_dict.items():
                epoches = [i for i in range(len(list))]
                plt_label = "user-"+str(user)+"-"+key
                if key == item : plt.plot(epoches, list, label=plt_label)
                # if key == "loss": plt_loss.plot(epoches, list, label=plt_label)

        plt.legend()
        rlr_info = ""
        if rlr == True: rlr_info = " using RLR "
        info = "Alpha="+ str(alpha) + ",User = "+ str(user_num)
        title = "Local Trainer "+ item + rlr_info + " " +info
        plt.title(title)
        plt.xlabel("Epoch")
        plt.ylabel(item)
        plt.savefig(title+".jpg")
        plt.close()


def train(spliter, alpha, user_num, global_rounds, local_epoches, batch_size, rlr = False,  log = True):
    avg_global_test_acc , avg_global_test_loss = 0 , 0
    dataset_dict = spliter.get_dataset_dict()
    label_partition_dict = spliter.get_label_partition_dict()
    dataset_size = spliter.get_complete_dataset_size()
    dataset_proportions = spliter.get_train_dataset_proportions()
    global_model = FcNet()
    # plot_partitions(user_num=user_num, label_partition_dict=label_partition_dict,
    #                 alpha=alpha)
    global_acc_loss_dict = {}
    local_acc_loss_dict = {}

    for user_index in range(user_num):
        local_acc_loss_dict[user_index] = {"acc": [], "loss": []}

    for round_idx in range(global_rounds):
        local_weights = []
        for user_index in range(user_num):
            local_dataset_size = dataset_dict[user_index]['train'].X_.shape[0]
            if log == True:
                print("_________________Local Trainer w/ size: ", local_dataset_size, "/",
                      dataset_size, "_________________________")
            model_weights, acc, loss = local_trainer(dataset=dataset_dict[user_index]['train'],
                                                model=copy.deepcopy(global_model),
                                                global_round=round_idx,
                                                local_epoch=local_epoches,
                                                batch_size = batch_size,
                                                log = log)
            local_weights.append(copy.deepcopy(model_weights))
            local_acc_loss_dict[user_index]["acc"].append(acc)
            local_acc_loss_dict[user_index]["loss"].append(loss)

        initial_global_weight = global_model.state_dict()
        user_index_sign_dict = {}
        i = 0
        for local_weight in local_weights:
            sign = \
                ratio_of_sign(w_global=initial_global_weight,
                          w_local=local_weight)
            user_index_sign_dict[i]=sign
            print( "Sign Number of User ", i, " : ", sign )
            i+=1

        model_params_num = sum(p.numel() for p in global_model.parameters()
                               if p.requires_grad)

        #Use Robust Learning Rate if true
        if rlr == True:
            # threshold = int(model_params_num * 0.05)
            threshold = 30
            print("Sign Number's Threshold : ", threshold)
            new_local_weights = []
            new_dataset_proportions = []
            for user_index, sign_dict in user_index_sign_dict.items():
                if abs(sign_dict['total']) > threshold :
                    new_local_weights.append(local_weights[user_index])
                    new_dataset_proportions.append(dataset_proportions[user_index])

            if len(new_local_weights) != 0:
                global_weight = weighted_average_weights(new_local_weights, new_dataset_proportions)
            else:
                global_weight = initial_global_weight
        else:
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
    plot_partitions(user_num=user_num,
                    label_partition_dict=label_partition_dict,
                    alpha=alpha,
                    rlr=rlr)
    plot_global_acc_loss(global_acc_loss_dict=global_acc_loss_dict,
                         user_num= user_num,
                         alpha=alpha,
                         rlr=rlr)

    plot_local_acc_loss(local_acc_loss_dict=local_acc_loss_dict,
                        user_num= user_num,
                        alpha= alpha,
                        rlr=rlr)

if __name__ == '__main__':
    alpha = 0.1
    user_num = 5
    global_rounds = 30
    local_epoches = 5
    batch_size = 64

    spliter = \
    DatasetSplitByDirichletPartition(file_path=config.data_path,
                                       alpha=alpha,
                                       user_num=user_num,
                                       train_ratio=.7)

    train(
            spliter = spliter,
            alpha = alpha,
            user_num = user_num,
            global_rounds = global_rounds,
            local_epoches = local_epoches,
            batch_size = batch_size,
            rlr = False,
            log=True)