#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.nn.init import normal_ as normal
from torchsummary import summary

class SurfaceMLPModel(nn.Module):
    def __init__(self, li=70, l1=128, l2=512, lo=1000):
        super(SurfaceMLPModel, self).__init__()

        self.fc_1 = nn.Linear(li, l1)
        normal(self.fc_1.weight)
        self.fc_2 = nn.Linear(l1, l2)
        normal(self.fc_2.weight)

        self.dropout_1 = nn.Dropout(0.5)

        self.output_layer = nn.Linear(l2, lo)
        normal(self.output_layer.weight)

        self.fc_activation = nn.ReLU()

        self.fc_bn1 = nn.BatchNorm1d(l1)
        self.fc_bn2 = nn.BatchNorm1d(l2)

    def forward(self, x):  # 216x216,1

        f1 = self.fc_1(x)
        f1 = self.fc_bn1(f1)
        fc_a1 = self.fc_activation(f1)

        fc_a1 = self.dropout_1(fc_a1)

        f2 = self.fc_2(fc_a1)
        f2 = self.fc_bn2(f2)
        fc_a2 = self.fc_activation(f2)

        output = self.output_layer(fc_a2)

        return output

