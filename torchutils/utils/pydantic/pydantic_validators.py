from typing import Optional, Any, Callable
import numpy as np
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def validate_torch_dataloader(other: Any) -> DataLoader:
    if isinstance(other, DataLoader):
        return other
    else:
        raise ValueError()


def validate_function(other: Any) -> Callable[[Any, Any], Any]:
    if callable(other):
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


def validate_nn_optimizer(other) -> Optional[Optimizer]:
    if isinstance(other, Optimizer):
        return other
    else:
        raise ValueError()


def validate_nn_scheduler(other) -> Optional[_LRScheduler]:
    if isinstance(other, _LRScheduler):
        return other
    else:
        raise ValueError()
