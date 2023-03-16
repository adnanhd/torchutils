import inspect
from torchutils.utils import(
    string_to_criterion_class,
    string_to_optimizer_class,
    string_to_scheduler_class,
    string_to_functionals,
    obtain_registered_kwargs,
    optimizer_class_to_string,
    scheduler_class_to_string
)
from typing import Optional, Any, Callable
import numpy as np
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


def validate_torch_dataloader(other: Any) -> DataLoader:
    if isinstance(other, DataLoader):
        return other
    else:
        raise ValueError()


def validate_torch_dataset(other: Any) -> Dataset:
    if isinstance(other, Dataset):
        return other
    else:
        raise ValueError()


def validate_np_scalar(dtype) -> np.generic:
    if isinstance(dtype, np.generic):
        return dtype
    else:
        raise ValueError()


def validate_np_array(arr) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    else:
        raise ValueError()


def validate_torch_tensor(arr) -> Tensor:
    if isinstance(arr, Tensor):
        return arr
    else:
        raise ValueError()


def validate_nn_module(other: Any) -> Optional[Module]:
    if isinstance(other, Module):
        return other
    else:
        raise ValueError()


def validate_criterion(other: Any) -> Callable[[Any, Any], Any]:
    if callable(other):
        return other
    elif isinstance(other, str):
        if other in string_to_criterion_class:
            return string_to_criterion_class[other]
        else:
            raise ValueError(f"{other} not registered criterion")
    else:
        raise ValueError()


def validate_nn_optimizer(other) -> Optional[Optimizer]:
    if inspect.isclass(other) and Optimizer.__subclasscheck__(other):
        return other
    elif (not inspect.isclass(other)) and \
            Optimizer.__subclasscheck__(other.__class__):
        return other
    elif not isinstance(other, str):
        raise ValueError()
    elif other in optimizer_class_to_string.values():
        return string_to_optimizer_class[other]
    else:
        raise ValueError(f"{other} not registered Optimizer")


def validate_nn_scheduler(other) -> Optional[_LRScheduler]:
    if inspect.isclass(other) and \
            (_LRScheduler.__subclasscheck__(other) or
             other is ReduceLROnPlateau):
        return other
    elif (not inspect.isclass(other)) and \
            (_LRScheduler.__subclasscheck__(other.__class__) or
             isinstance(other, ReduceLROnPlateau)):
        return other
    elif not isinstance(other, str):
        raise ValueError()
    elif other in scheduler_class_to_string.values():
        return string_to_scheduler_class[other]
    else:
        raise ValueError(f"{other} not registered Scheduler")
