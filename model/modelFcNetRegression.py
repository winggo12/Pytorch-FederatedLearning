from __future__ import print_function
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional

class Threshold(nn.Module):
    def __init__(self):
        super(Threshold, self).__init__()
        self.threshold = nn.Parameter(torch.full(size=(1,), fill_value=0.5))

    def forward(self, x):
        results = torch.Tensor([])
        for tensor in x:
            decimal_value = tensor - torch.trunc(tensor)
            if decimal_value > self.threshold:
                result = torch.round(tensor)
                results = torch.cat((result, results), dim=0)
            else:
                result = torch.floor(tensor)
                results = torch.cat((result, results), dim=0)

        results = results[:, None]

        return results

class FcNetRegression(nn.Module):
    def __init__(self):
        super(FcNetRegression, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self,x):
        x = functional.relu(self.fc1(x))
        x = functional.relu(self.fc2(x))
        return x

class FCNetRegressionThreshold(nn.Module):
    def __init__(self):
        super(FCNetRegressionThreshold, self).__init__()
        self.fcnetregression = FcNetRegression()
        self.threshold = Threshold()

    def forward(self, x):
        x = self.fcnetregression(x)
        x = self.threshold(x)
        return x