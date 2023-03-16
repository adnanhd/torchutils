#!/usr/bin/env python3.6
import torch.nn as nn
from .mlp import FeedForward
from .cnn import Convolution, Encoder, Decoder
from .autoencoders import AutoencoderCNN, AutoencoderMLP
from .utils import TrainerModel
#from .ausm_model import CFD_CNN as AUSM_CFD_CNN


def BezierMLP(li=70, l1=256, l2=128, l3=64, lo=12):
    return FeedForward(li, l1, l2, l3, lo)


def SurfaceMLP(li=70, l1=128, l2=512, lo=1000):
    return FeedForward(li, l1, l2, lo)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = tuple(args)

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)

    def __repr__(self):
        return self.__class__.__name__ + str((None,) + self.shape)


__version__ = "1.0.0"
