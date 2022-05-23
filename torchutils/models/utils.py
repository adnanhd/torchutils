import torch.nn as nn


def _add_modules(self, modules, layer_num=None):
    if layer_num is None:
        for key, value in modules.items():
            self.add_module(key, value)
    else:
        for key, value in modules.items():
            self.add_module(f'{key}{layer_num}', value)

def _argument(*args, **kwargs):
    return {'args': args, 'kwargs': kwargs}

def _add_last_layer(self, output, choices=(None, ), *args, **kwargs):
    assert (output in choices) or isinstance(output, nn.Module)
    
    if output == 'linear':
        self.add_module('linear', nn.Linear(*args, **kwargs))
    elif output == 'dropout':
        self.add_module('dropout', nn.Dropout(*args, **kwargs))
    elif output == 'sigmoid':
        self.add_module('sigmoid', nn.Sigmoid(*args, **kwargs))
    elif isinstance(output, nn.Module):
        self.add_module('submodule', output)

