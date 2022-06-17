import torch
import numpy as np
from torch.functional import F
import torch.optim as optim
import torch.nn.modules.loss as criterion
import torch.optim.lr_scheduler as sched


string_to_optimizer_class = {
    name: optimizer_class
    for name, optimizer_class in vars(optim).items()
    if hasattr(optimizer_class, 'mro')
    and optim.Optimizer in optimizer_class.mro()
}

string_to_criterion_class = {
    name: criterion_class
    for name, criterion_class in vars(criterion).items()
    if hasattr(criterion_class, 'mro')
    and criterion._Loss in criterion_class.mro()
    and criterion_class is not criterion._Loss
}

string_to_functionals = {
    name: loss_func
    for name, loss_func in vars(F).items()
    if callable(loss_func)
}

string_to_scheduler_class = {
    name: scheduler_class
    for name, scheduler_class in vars(sched).items()
    if hasattr(scheduler_class, 'mro')
    and sched._LRScheduler in scheduler_class.mro()
    and scheduler_class is not sched._LRScheduler
}

# initialize default types
string_to_types = {
    'string': str, 'str': str,
    'integer': int, 'int': int,
    'boolean': bool, 'bool': bool,
    'float': float, 'bytes': bytes
}

# Add numpy types
string_to_types.update({
    f'np.{name}': NpType
    for name, NpType in np.typeDict.items()
})

# Add tensor types
string_to_types.update({
    f'torch.{name}': tensorType
    for name, tensorType in vars(torch).items()
    if isinstance(tensorType, torch.dtype)
})
