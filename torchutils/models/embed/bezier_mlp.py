#!/usr/bin/env python3
import torch
import torch.nn as nn
from torchsummary import summary



class BezierMLPModel(nn.Module):
    def __init__(self, li=70, l1=256, l2=128, l3=64, lo=12):
        super(BezierMLPModel, self).__init__()

        self.fc_1 = nn.Linear(li, l1)
        self.fc_2 = nn.Linear(l1, l2)
        self.fc_3 = nn.Linear(l2, l3)

        self.output_layer = nn.Linear(l3, lo)

        self.fc_activation = nn.ReLU()

        self.fc_bn1 = nn.BatchNorm1d(l1)
        self.fc_bn2 = nn.BatchNorm1d(l2)
        self.fc_bn3 = nn.BatchNorm1d(l3)

    def forward(self, x):  # 216x216,1

        f1 = self.fc_1(x)
        f1 = self.fc_bn1(f1)
        fc_a1 = self.fc_activation(f1)

        f2 = self.fc_2(fc_a1)
        f2 = self.fc_bn2(f2)
        fc_a2 = self.fc_activation(f2)

        f3 = self.fc_3(fc_a2)
        f3 = self.fc_bn3(f3)
        fc_a3 = self.fc_activation(f3)

        output = self.output_layer(fc_a3)

        return output


if __name__ == '__main__':
    model = BezierMLPModel()
