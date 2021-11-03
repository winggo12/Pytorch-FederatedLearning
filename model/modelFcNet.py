from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional

class FcNet(nn.Module):
    def __init__(self):
        super(FcNet, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        # self.fc2 = nn.Linear(18, 36)
        # self.fc3 = nn.Linear(36, 54)
        # self.fc4 = nn.Linear(54, 36)
        # self.fc5 = nn.Linear(36, 18)
        # self.fc6 = nn.Linear(18, 10)
        self.fc7 = nn.Linear(16, 10)

        # For Regression
        # self.fc6 = nn.Linear(18, 1)

        self.dropout = nn.Dropout(0.25)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        # nn.init.xavier_uniform_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)
        # nn.init.xavier_uniform_(self.fc3.weight)
        # nn.init.zeros_(self.fc3.bias)
        # nn.init.xavier_uniform_(self.fc4.weight)
        # nn.init.zeros_(self.fc4.bias)
        # nn.init.xavier_uniform_(self.fc5.weight)
        # nn.init.zeros_(self.fc5.bias)
        nn.init.xavier_uniform_(self.fc7.weight)
        nn.init.zeros_(self.fc7.bias)

    def forward(self,x):
        # x = x.view(x.size(0), -1)
        x = functional.leaky_relu(self.fc1(x))
        # x = functional.leaky_relu(self.fc2(x))
        # x = functional.leaky_relu(self.fc3(x))
        # x = functional.leaky_relu(self.fc4(x))
        # x = functional.leaky_relu(self.fc5(x))
        x = self.fc7(x)
        return x