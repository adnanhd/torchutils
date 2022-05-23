#!/usr/bin/env python3

import torch
import torch.nn as nn
from .utils import _add_modules, _add_last_layer


class Perceptron(torch.nn.Sequential):
    def __init__(self, in_channels, out_channels, *args, epiloques={}, proloques={}, **kwargs):
        super(Perceptron, self).__init__()
        _add_modules(self, proloques)
        self.add_module('hidden', nn.Linear(in_channels, out_channels))
        _add_modules(self, epiloques)


class FeedForward(torch.nn.Sequential):
    def __init__(self, *layers, activation=nn.ReLU(), norm=False, output=None, output_args=[], output_kwargs={}, **kwargs):
        assert len(layers) >= 2
        super(FeedForward, self).__init__()

        for i in range(1, len(layers) - 1):
            epiloques = {'relu': activation}
            fc_layer = nn.Linear(layers[i-1], layers[i])
            self.add_module(f'hidden{i}', fc_layer)
            if norm:
                self.add_module(f'bnorm{i}', nn.BatchNorm1d(layers[i]))            
            if activation is not None:
                self.add_module(f'relu{i}', activation)

        self.add_module('output', nn.Linear(layers[-2], layers[-1]))
        _add_last_layer(self, output, (None, 'dropout', 'sigmoid'), *output_args, **output_kwargs)

