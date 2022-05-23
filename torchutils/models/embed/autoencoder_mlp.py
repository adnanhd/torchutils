#!/usr/bin/env python3
import torch
import torch.nn as nn
from .mlp import FullyConnectedBlock


class AutoencoderMLPModel(nn.Module):
    def __init__(self, li=3136, l1=128, l2=128, l3=128, lo=1000, init_weight=None):
        super(AutoencoderMLPModel, self).__init__()
        self.li = li

        ### ENCODER
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=4,
                                kernel_size=4,
                                stride=1,
                                padding=1)

        self.conv_2 = nn.Conv2d(in_channels=4,
                                out_channels=16,
                                kernel_size=4,
                                stride=1,
                                padding=1)

        self.conv_3 = nn.Conv2d(in_channels=16,
                                out_channels=64,
                                kernel_size=4,
                                stride=1,
                                padding=1)

        self.bn1 = nn.BatchNorm2d(4)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.pool3 = nn.MaxPool2d(3, 3)
        self.activation = nn.ReLU()
        
        ### Parameter part
        self.features = nn.Linear(li, 16)
        self.fcb = FullyConnectedBlock(li, l1, l2, l3, lo, activation=nn.ReLU())

    def forward(self, x):                     ## [None, 1, 216, 216]

        c1 = self.conv_1(x)
        c1 = self.bn1(c1)
        a1 = self.pool3(self.activation(c1))  ## [None, 4, 73, 73]

        c2 = self.conv_2(a1)
        c2 = self.bn2(c2)
        a2 = self.pool3(self.activation(c2))  ## [None, 16, 23, 23]

        c3 = self.conv_3(a2)
        c3 = self.bn3(c3)
        a3 = self.pool3(self.activation(c3))  ## [None, 64, 7, 7]

        a3 = a3.view(-1, self.li)
        a3 = self.features(a3)

        return self.fcb(a3)

