import pydantic
from abc import ABC
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional, Any


from .pydantic_validators import validate_np_scalar
class NpScalarType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod  
    def __get_validators__(cls):  
        yield validate_np_scalar


from .pydantic_validators import validate_np_array
from .pydantic_validators import validate_torch_tensor
class NpTorchType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod  
    def __get_validators__(cls):  
        yield validate_np_array or validate_torch_tensor


from .pydantic_validators import validate_torch_dataset
class DatasetType(ABC):

    @classmethod
    def __get_validators__(cls):
        yield validate_torch_dataset


from .pydantic_validators import validate_torch_dataloader
class DataLoaderType(ABC):
    @classmethod
    def __get_validators__(cls):
        yield validate_torch_dataloader


from .pydantic_validators import validate_nn_module
class ModuleType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod  
    def __get_validators__(cls):  
        yield validate_nn_module


class LossType(ModuleType):
    pass

class FunctionType(ABC):
    #TODO: implement this class
    pass


from .pydantic_validators import validate_nn_optimizer
class OptimizerType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """

    @classmethod  
    def __get_validators__(cls):  
        yield validate_nn_optimizer
    

from .pydantic_validators import validate_nn_scheduler
class SchedulerType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """

    @classmethod  
    def __get_validators__(cls):  
        yield validate_nn_scheduler

