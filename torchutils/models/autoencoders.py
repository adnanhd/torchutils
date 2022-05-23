import torch
import torch.nn as nn
from .mlp import FeedForward
from .cnn import Convolution

class AutoencoderCNN(nn.Module):
    def __init__(self):
        super(AutoencoderCNN, self).__init__()

        ### ENCODER
        self.cnn1 = Convolution(1, 32, 32, kernel_size=4, maxpool=(3,3))
        self.cnn2 = Convolution(32, 64, 128, kernel_size=4, maxpool=(2,2))
        self.fcn1 = FeedForward(128, 100, 100, 16)
        self.fcn2 = FeedForward(16, 100, 100, 140)

    def forward(self, x):  ##216x216,1
        x = self.cnn1(x)
        x = self.cnn2(x)

        x = x.view(-1, 128)
        
        x = self.fcn1(x)
        x = self.fcn2(x)
        
        return x

class AutoencoderMLP(nn.Module):
    def __init__(self, li=3136, l1=128, l2=128, l3=128, lo=1000, init_weight=None):
        super(AutoencoderMLP, self).__init__()
        self.li = li

        self.conv_block = Convolution(1, 4, 16, 64, kernel_size=4, maxpool=(3, 3), activation=nn.ReLU()) 
        self.features = nn.Linear(li, 16)
        self.fcb = FeedForward(16, l1, l2, l3, lo, activation=nn.ReLU())

    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(-1, self.li)
        x = self.features(x)

        return self.fcb(x)
