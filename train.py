import torch
import torch.nn as nn
from model.modelFcNet import FcNet
from datasetProcessing.dataset import TheDataset

batch_size = 16
epoches = 200
iteration = 0

running_loss = 0
running_corrects = 0
running_loss_per_itr = 0
running_corrects_per_itr = 0

bank_dataset = TheDataset("./data/BankChurners_normalized.csv")
train_loader = torch.utils.data.DataLoader(bank_dataset, batch_size=batch_size, shuffle=True)

model = FcNet()
params_to_update = model.parameters()

decayRate = 0.96
loss_func = nn.CrossEntropyLoss()
#For Regression
# loss_func = nn.MSELoss()

optimizer = torch.optim.SGD(params_to_update, lr=0.0001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

for epoch in range(epoches):

    for inputs , labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        #For Classification
        _, preds = torch.max(outputs, 1)
        _, ground_truth = torch.max(labels, 1)

        #For Regression
        # preds = torch.round(outputs)
        # ground_truth = labels

        running_loss += loss.item()
        running_loss += loss.item() * inputs.size(0)
        running_loss_per_itr += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds.data == ground_truth.data)
        running_corrects_per_itr += torch.sum(preds.data == ground_truth.data)

        if iteration % 100 == 0:
            acc = running_corrects_per_itr / (100 * batch_size)
            loss = running_loss_per_itr / (100 * batch_size)
            running_corrects_per_itr = 0
            running_loss_per_itr = 0
            print("Iteration:", iteration, " Acc: ", acc, " Loss: ", loss)

        # if iteration % 1000 == 0:
        #     scheduler.step()

        iteration += 1



    # print(loss.item())