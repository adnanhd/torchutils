import torch
import torch.nn as nn


class AutoencoderCNNModel(nn.Module):
    def __init__(self):
        super(AutoencoderCNNModel, self).__init__()

        ### ENCODER
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=32,
                                kernel_size=4,
                                stride=1,
                                padding=1)

        self.conv_2 = nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=4,
                                stride=1,
                                padding=1)

        self.conv_3 = nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=4,
                                stride=1,
                                padding=1)
        self.conv_4 = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=4,
                                stride=1,
                                padding=1)
        self.conv_5 = nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=4,
                                stride=1,
                                padding=1)

        self.fc_1 = nn.Linear(128, 100)
        self.fc_2 = nn.Linear(100, 100)
        self.fc_3 = nn.Linear(100, 100)

        self.features = nn.Linear(100, 16)

        self.fc_4 = nn.Linear(16, 100)
        self.fc_5 = nn.Linear(100, 100)
        self.fc_6 = nn.Linear(100, 100)

        self.output_layer = nn.Linear(100, 140)

        self.activation = nn.ReLU()
        self.fc_activation = nn.Tanh()

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(128)

        self.fc_bn1 = nn.BatchNorm1d(100)
        self.fc_bn2 = nn.BatchNorm1d(100)
        self.fc_bn3 = nn.BatchNorm1d(100)
        self.fc_bnf = nn.BatchNorm1d(16)
        self.fc_bn4 = nn.BatchNorm1d(100)
        self.fc_bn5 = nn.BatchNorm1d(100)
        self.fc_bn6 = nn.BatchNorm1d(100)
        self.fc_bno = nn.BatchNorm1d(140)

        self.pool3 = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

    def forward(self, x):  ##216x216,1

        c1 = self.conv_1(x)
        c1 = self.bn1(c1)
        a1 = self.pool3(self.activation(c1))  ## 71x71

        c2 = self.conv_2(a1)
        c2 = self.bn2(c2)
        a2 = self.pool3(self.activation(c2))  ## 23x23

        c3 = self.conv_3(a2)
        c3 = self.bn3(c3)
        a3 = self.pool3(self.activation(c3))  ## 7x7

        c4 = self.conv_4(a3)
        c4 = self.bn4(c4)
        a4 = self.pool2(self.activation(c4))  ## 3x3

        c5 = self.conv_5(a4)
        c5 = self.bn5(c5)
        a5 = self.pool2(self.activation(c5))  ## 1x1

        a5 = a5.view(-1, 128)

        f1 = self.fc_1(a5)
        f1 = self.fc_bn1(f1)
        fc_a1 = self.fc_activation(f1)

        f2 = self.fc_2(fc_a1)
        f2 = self.fc_bn2(f2)
        fc_a2 = self.fc_activation(f2)

        f3 = self.fc_3(fc_a2)
        f3 = self.fc_bn3(f3)
        fc_a3 = self.fc_activation(f3)

        features = self.features(fc_a3)
        features = self.fc_bnf(features)
        features_a = self.fc_activation(features)

        f4 = self.fc_4(features_a)
        f4 = self.fc_bn4(f4)
        fc_a4 = self.fc_activation(f4)

        f5 = self.fc_5(fc_a4)
        f5 = self.fc_bn5(f5)
        fc_a5 = self.fc_activation(f5)

        f6 = self.fc_6(fc_a5)
        f6 = self.fc_bn6(f6)
        fc_a6 = self.fc_activation(f6)

        output = self.output_layer(fc_a6)

        return output


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#net = AutoencoderCNNModel().to(device)
# summary(net, (1,216,216))
