#!/usr/bin/env python3
import torch
import torch.nn as nn
from .utils import _add_modules, _add_last_layer

class Encoder(torch.nn.Sequential):
    def __init__(self, *layers, kernel_size, activation=nn.ReLU(), norm=False, pool=None, output=None, output_args=[], output_kwargs={}, **kwargs):
        assert len(layers) >= 2
        super(Encoder, self).__init__()

        for i in range(1, len(layers) - 1):
            self.add_module(
                    f'conv{i}',
                nn.Conv2d(
                    in_channels=layers[i-1],
                    out_channels=layers[i],
                    kernel_size=kernel_size,
                    **kwargs)
            )

            if norm:
                self.add_module(f'bnorm{i}', nn.BatchNorm2d(layers[i]))

            if activation is not None:
                self.add_module(f'activ{i}', activation)

            if pool is not None:
                sefl.add_module(f'pool{i}', pool)

        self.add_module(f'output', nn.Conv2d(layers[-2], layers[-1], kernel_size, **kwargs))
        _add_last_layer(self, output, (None, 'sigmoid', 'linear'), *output_args, **output_kwargs)


class Decoder(torch.nn.Sequential):
    def __init__(self, *layers, kernel_size, activation=nn.ReLU(), output=None, **kwargs):
        assert len(layers) >= 2
        super(Decoder, self).__init__()

        for i in range(1, len(layers) - 1):
            self.add_module(
                    f'deconv{i}',
                nn.ConvTranspose2d(
                    in_channels=layers[i-1],
                    out_channels=layers[i],
                    kernel_size=kernel_size,
                    **kwargs)
            )

            if activation is not None:
                self.add_module(f'activ{i}', activation)
        
        self.add_module(f'output', nn.ConvTranspose2d(layers[-2], layers[-1], kernel_size, **kwargs))


def Convolution(*layers, kernel_size, multiple=1, **kwargs):
    last_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layers[-2] * multiple, layers[-1]),
            nn.Sigmoid()
    )
    return Encoder(*layers[:-1], kernel_size=kernel_size, output=last_layer, **kwargs)

