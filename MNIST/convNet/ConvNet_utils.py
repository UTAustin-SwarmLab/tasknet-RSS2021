import torch
import torch.nn as nn
import numpy as np

import sys, os

# Convolutional neural network (two convolutional layers)
# this was used for MNIST experiments
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(8 * 8 * 64, 1000)
        self.fc2 = nn.Linear(1000, self.num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out, None