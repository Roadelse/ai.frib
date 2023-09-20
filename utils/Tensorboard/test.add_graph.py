#!/usr/bin/env python
# coding: utf-8

import sys
import time

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

net = LeNet()



writer = SummaryWriter(comment='test_add_graph',filename_suffix="_test_add_graph_suffix")


a = np.random.normal(1, 1, (177, 1, 28, 28))

writer.add_graph(net, torch.Tensor(a))
writer.close()

