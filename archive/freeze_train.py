import copy
import torch
import torch.nn as nn
from model.modelFcNet import FcNet
from model.modelFcNetRegression import FcNetRegression, FCNetRegressionThreshold
from dataset_processing.dataset import TheDataset


batch_size = 32
epoches = 2000
iteration = 0

running_loss = 0
running_corrects = 0
running_loss_per_itr = 0
running_corrects_per_itr = 0

bank_dataset = TheDataset("../data/BankChurners_normalized.csv")
train_loader = torch.utils.data.DataLoader(bank_dataset, batch_size=batch_size, shuffle=True)

model_path = "../saved/model_regression.pth"
model = FCNetRegressionThreshold()
model.fcnetregression.load_state_dict(torch.load(model_path))

params_to_update = model.parameters()
# params_to_freeze = ["FcReluNet"]
params_to_freeze = ["fcnetregression"]
print("Params to learn:")

params_to_update = []
for name, param in model.named_parameters():
    for param_freeze in params_to_freeze:
        if param_freeze not in name:
            params_to_update.append(param)
            print("\t", name)

decayRate = 0.96
loss_func = nn.CrossEntropyLoss()
#For Regression
loss_func = nn.MSELoss()

optimizer = torch.optim.AdamW(params_to_update, lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

for epoch in range(epoches):

    for inputs , labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()

        #For Classification
        # _, preds = torch.max(outputs, 1)
        # _, ground_truth = torch.max(labels, 1)

        #For Regression
        preds = torch.round(outputs)
        # preds = outputs
        ground_truth = labels

        running_loss += loss.item()
        running_loss += loss.item() * inputs.size(0)
        running_loss_per_itr += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds.data == ground_truth.data)
        running_corrects_per_itr += torch.sum(preds.data == ground_truth.data)

        if iteration % 100 == 0 and iteration != 0 :
            acc = running_corrects_per_itr / (100 * batch_size)
            loss = running_loss_per_itr / (100 * batch_size)
            running_corrects_per_itr = 0
            running_loss_per_itr = 0
            print("Iteration:", iteration, " Acc: ", acc, " Loss: ", loss)



        iteration += 1

#Final Loss :
final_loss = running_loss/(iteration * batch_size)
final_acc = running_corrects/(iteration * batch_size)
print("Final Acc: ", final_acc, " Final Loss: ", final_loss)

