from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional
from config import config

class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc7 = nn.Linear(16, config.num_class)
        self.dropout = nn.Dropout(0.25)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc7.weight)
        nn.init.zeros_(self.fc7.bias)

    def forward(self,x):
        x = functional.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc7(x)

        return x

class DeeperFcNet(nn.Module):
    def __init__(self):
        super(DeeperFcNet, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.fc7 = nn.Linear(32, 16)
        self.fc8 = nn.Linear(16, 10)

        self.dropout = nn.Dropout(0.2)


    def forward(self,x):
        x = functional.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = functional.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = functional.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = functional.leaky_relu(self.fc4(x))
        x = self.dropout(x)
        x = functional.leaky_relu(self.fc5(x))
        x = self.dropout(x)
        x = functional.leaky_relu(self.fc6(x))
        x = self.dropout(x)
        x = functional.leaky_relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        return x