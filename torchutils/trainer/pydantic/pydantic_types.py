import pydantic
from abc import ABC
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Any
from .pydantic_validators import validate_torch_dataset
from .pydantic_validators import validate_torch_dataloader


class DatasetType(ABC):
    @classmethod
    def __get_validators__(cls):
        yield validate_torch_dataset


class DataLoaderType(ABC):
    @classmethod
    def __get_validators__(cls):
        yield validate_torch_dataloader


class ModuleType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod  
    def __get_validators__(cls):  
        yield cls.validate_nn_module

    @classmethod
    def validate_nn_module(cls, other: Any) -> Optional[Module]:
        if isinstance(other, Module): return other
        else: raise ValueError(f'{other} is not an instance of a class inherited from {Module.__qualname__}')


class LossType(ModuleType):
    pass

class FunctionType(ABC):
    #TODO: implement this class
    pass


class OptimizerType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod  
    def __get_validators__(cls):  
        yield cls.validate_nn_optimizer

    @classmethod
    def validate_nn_optimizer(cls, other) -> Optional[Optimizer]:
        if isinstance(other, Optimizer): return other
        else: raise ValueError(f'{other} is not an instance of a class inherited from {Optimizer.__qualname__}')

class SchedulerType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod  
    def __get_validators__(cls):  
        yield cls.validate_nn_scheduler

    @classmethod
    def validate_nn_scheduler(cls, other) -> Optional[_LRScheduler]:
        if isinstance(other, _LRScheduler): return other
        else: raise ValueError(f'{other} is not an instance of a class inherited from {_LRScheduler.__qualname__}')

