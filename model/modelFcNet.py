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

        # For Regression
        # self.fc6 = nn.Linear(18, 1)

        self.dropout = nn.Dropout(0.25)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc7.weight)
        nn.init.zeros_(self.fc7.bias)

    def forward(self,x):
        x = functional.leaky_relu(self.fc1(x))
        x = self.fc7(x)
        return x