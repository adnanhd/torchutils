import torch
import numpy as np

from torch.functional import F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.modules.loss as criterion


string_to_optimizer_class = {
    optimizer_class.__name__: optimizer_class
    for optimizer_class in Optimizer.__subclasses__()
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

# string_to_scheduler_class = {
#     name: scheduler_class
#     for name, scheduler_class in vars(sched).items()
#     if hasattr(scheduler_class, 'mro')
#     and Scheduler in scheduler_class.mro()
#     and scheduler_class is not Scheduler
#     or scheduler_class is sched.ReduceLROnPlateau
# }

string_to_scheduler_class = {
    scheduler.__name__: scheduler
    for scheduler in Scheduler.__subclasses__()
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


def reverse_dict(d: dict):
    return {v: k for k, v in d.items()}


# reverse types
types_to_string = reverse_dict(string_to_types)
functionals_to_string = reverse_dict(string_to_functionals)
criterion_class_to_string = reverse_dict(string_to_criterion_class)
optimizer_class_to_string = reverse_dict(string_to_optimizer_class)
scheduler_class_to_string = reverse_dict(string_to_scheduler_class)

# registary modules


def _register_criterion(criterion):
    global string_to_criterion_class
    global criterion_class_to_string
    string_to_criterion_class[criterion.__name__] = criterion
    criterion_class_to_string[criterion] = criterion.__name__


def _register_optimizer(optimizer: Optimizer):
    global string_to_optimizer_class
    global optimizer_class_to_string
    string_to_optimizer_class[optimizer.__name__] = optimizer
    optimizer_class_to_string[optimizer] = optimizer.__name__


def _register_scheduler(scheduler: Scheduler):
    global string_to_scheduler_class
    global scheduler_class_to_string
    string_to_scheduler_class[scheduler.__name__] = scheduler
    scheduler_class_to_string[scheduler] = scheduler.__name__


_register_scheduler(ReduceLROnPlateau)
