import copy
import torch
import torch.nn as nn
from model.modelFcNet import FcNet
from model.modelFcNetRegression import FcNetRegression
from datasetProcessing.dataset import TheDataset

batch_size = 32
epoches = 200
iteration = 0

running_loss = 0
running_corrects = 0
running_loss_per_itr = 0
running_corrects_per_itr = 0

bank_train_dataset = TheDataset("./data/BankChurners_normalized.csv", split_ratio=0.9, train_or_test='train')
bank_test_dataset = TheDataset("./data/BankChurners_normalized.csv", split_ratio=0.9, train_or_test='test')

train_loader = torch.utils.data.DataLoader(bank_train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(bank_test_dataset, batch_size=batch_size, shuffle=True)

model = FcNet()
#For Regression
model = FcNetRegression()
params_to_update = model.parameters()

decayRate = 0.96
loss_func = nn.CrossEntropyLoss()
#For Regression
loss_func = nn.MSELoss()
optimizer = torch.optim.AdamW(params_to_update, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

class Status:
    def __init__(self,name,dataloader):
        self.name = name
        self.dataloader = dataloader
        self.iteration = 0
        self.running_corrects = 0
        self.running_corrects_per_itr = 0
        self.running_loss = 0
        self.running_loss_per_itr = 0

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
            # _, preds = torch.max(outputs, 1)
            # _, ground_truth = torch.max(labels, 1)

            #For Regression
            preds = torch.round(outputs)
            # preds = outputs
            ground_truth = labels

            status.running_loss += loss.item()
            status.running_loss += loss.item() * inputs.size(0)
            status.running_loss_per_itr += loss.item() * inputs.size(0)
            status.running_corrects += torch.sum(preds.data == ground_truth.data)
            status.running_corrects_per_itr += torch.sum(preds.data == ground_truth.data)

            if status.iteration % 1000 == 0 and status.iteration!=0:
                acc = status.running_corrects_per_itr / (1000 * batch_size)
                loss = status.running_loss_per_itr / (1000 * batch_size)
                status.running_corrects_per_itr = 0
                status.running_loss_per_itr = 0
                print("Stage: ", status.name ," Iteration:", status.iteration, " Acc: ", acc, " Loss: ", loss)

            status.iteration += 1

#Result :
# for status in [trainstatus, teststatus]:
#     final_loss = status.running_loss/(status.iteration * batch_size)
#     final_acc = status.running_corrects/(status.iteration * batch_size)
#     print("______________________________")
#     print("Final Stage: ", status.name ," Acc: ", final_loss, "  Loss: ", final_loss)


# Save Model:
saved_model = copy.deepcopy(model.state_dict())
torch.save(saved_model, "./saved/model_regression.pth")