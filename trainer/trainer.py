import numpy as np
import torch
import torch.nn as nn
from ensemble_learning.sklearn_utils import display_result

def local_trainer(dataset, model, global_round, local_epoch, batch_size, log=True):
    iteration = 0
    running_corrects, running_corrects_per_itr = 0 , 0
    running_loss, running_loss_per_itr = 0 , 0
    final_acc, final_loss = 0, 0
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset_size = dataset.X_.shape[0]
    report_iterations = int( int(dataset_size/batch_size)*local_epoch  * 0.99)
    test_preds, test_ground_truth = np.asarray([]), np.asarray([])

    for epoch in range(local_epoch):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            _, ground_truth = torch.max(labels, 1)

            test_preds = np.append(test_preds, preds.numpy())
            test_ground_truth = np.append(test_ground_truth, ground_truth.numpy())

            running_loss += loss.item() * inputs.size(0)
            running_loss_per_itr += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.data == ground_truth.data)
            running_corrects_per_itr += torch.sum(preds.data == ground_truth.data)

            if iteration == report_iterations and iteration!=0:
                acc = running_corrects_per_itr / (report_iterations * batch_size)
                loss = running_loss_per_itr / (report_iterations * batch_size)
                running_corrects_per_itr = 0
                running_loss_per_itr = 0
                final_acc = acc
                if log == True:
                    print('Stage: Train  Global Round {} Iteration: {} Acc: {}  Loss: {}'.format(global_round,iteration,acc,loss))
            iteration += 1

    # final_acc, per_class_acc, cm, cr = display_result(labels=test_ground_truth,
    #                                             predictions=test_preds,
    #                                             log=False)
    # print("Per-class acc: ", per_class_acc)
    final_loss = running_loss/(local_epoch*dataset_size)

    return model.state_dict(), final_acc, final_loss

def inference(dataset, model, batch_size, log=True):
    iteration = 0
    running_corrects, running_corrects_per_itr = 0, 0
    running_loss, running_loss_per_itr = 0, 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    # model.train()
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    test_preds, test_ground_truth = np.asarray([]), np.asarray([])

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        # loss.backward()
        # optimizer.step()
        _, preds = torch.max(outputs, 1)
        _, ground_truth = torch.max(labels, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds.data == ground_truth.data)
        test_preds = np.append(test_preds, preds.numpy())
        test_ground_truth = np.append(test_ground_truth, ground_truth.numpy())
        iteration += 1

    acc, per_class_acc, cm, cr = display_result(labels=test_ground_truth,
                                                predictions=test_preds,
                                                log=False)
    final_acc = running_corrects/(dataset.get_dataset_size())
    final_loss = running_loss / (dataset.get_dataset_size())

    return final_acc, final_loss, per_class_acc