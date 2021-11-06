import torch
import torch.nn as nn

def local_trainer(dataset, model, global_round, local_epoch, batch_size):
    iteration = 0
    running_corrects, running_corrects_per_itr = 0 , 0
    running_loss, running_loss_per_itr = 0 , 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(local_epoch):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            _, ground_truth = torch.max(labels, 1)
            # running_loss += loss.item()
            running_loss += loss.item() * inputs.size(0)
            running_loss_per_itr += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds.data == ground_truth.data)
            running_corrects_per_itr += torch.sum(preds.data == ground_truth.data)

            if iteration % 100 == 0 and iteration!=0:
                acc = running_corrects_per_itr / (100 * batch_size)
                loss = running_loss_per_itr / (100 * batch_size)
                running_corrects_per_itr = 0
                running_loss_per_itr = 0
                print('Stage: Train  Global Round {} Iteration: {} Acc: {}  Loss: {}'.format(global_round,iteration,acc,loss))
            iteration += 1

    final_loss = running_loss/(iteration)

    return model.state_dict(), final_loss

def inference(dataset, model, batch_size):
    iteration = 0
    running_corrects, running_corrects_per_itr = 0, 0
    running_loss, running_loss_per_itr = 0, 0

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_func(outputs, labels)

        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        _, ground_truth = torch.max(labels, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds.data == ground_truth.data)

        iteration += 1

    final_acc = running_corrects/(iteration*batch_size)
    final_loss = running_loss / (iteration*batch_size)


    return final_acc, final_loss