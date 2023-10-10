import typing
import torch.nn as nn

from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torchutils.utils import obtain_registered_kwargs

from .pydantic.types import NpTorchType
from .mappings import (
    string_to_criterion_class,
    string_to_optimizer_class,
    string_to_scheduler_class,
    string_to_functionals,
)


def __fetch_criterion(
    criterion: typing.Union[str, nn.Module,
                            typing.Callable[[NpTorchType, NpTorchType],
                                            NpTorchType]] = None,
    ** kwargs,
):
    if isinstance(criterion, str):
        if criterion in string_to_criterion_class:
            criterion_class = string_to_criterion_class[criterion]
        elif criterion in string_to_functionals:
            criterion_class = string_to_functionals[criterion]
        else:
            raise KeyError(
                f"{criterion} is not a registered function or Module"
            )
        criterion_params = obtain_registered_kwargs(
            criterion_class, kwargs)
        criterion = criterion_class(**criterion_params)
    return criterion


def __fetch_optimizer(
    model: nn.Module = None,
    optimizer: typing.Union[str, Optimizer] = None,
    ** kwargs,
):
    if isinstance(optimizer, str):
        if optimizer in string_to_optimizer_class:
            optimizer = string_to_optimizer_class[optimizer]
            params = obtain_registered_kwargs(optimizer, kwargs)
            optimizer = optimizer(model.parameters(), **params)
        else:
            raise KeyError(
                f"{optimizer} is not a registered Optimizer"
            )
    return optimizer


def __fetch_scheduler(
    optimizer: typing.Union[str, Optimizer] = None,
    scheduler: typing.Union[str, _LRScheduler, None] = None,
    ** kwargs,
):
    if isinstance(scheduler, str):
        if scheduler in string_to_scheduler_class:
            scheduler = string_to_scheduler_class[scheduler]
            params = obtain_registered_kwargs(scheduler, kwargs)
            scheduler = scheduler(optimizer, **params)
        else:
            raise KeyError(
                f"{scheduler} is not a registered Scheduler"
            )
    return scheduler


def __compile_modules(
    model: nn.Module,
    optimizer: typing.Union[str, Optimizer],
    scheduler: typing.Union[str, _LRScheduler, None],
    ** kwargs,
):
    __module_names__ = ('model', 'optimizer', 'scheduler')
    try:
        params = obtain_registered_kwargs(optimizer, kwargs)
        optimizer.add_param_groups({'params': model.parameters(), **params})
    except ValueError:
        # model params are already in the model
        pass  # i.e. model is not changed yet
    scheduler.optimizer = optimizer
    return tuple(map(vars().__getitem__, __module_names__))
