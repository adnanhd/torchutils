from .pydantic_validators import validate_function
from .pydantic_validators import validate_nn_scheduler
from .pydantic_validators import validate_nn_optimizer
from .pydantic_validators import validate_nn_module
from .pydantic_validators import validate_torch_dataloader
from .pydantic_validators import validate_torch_dataset
from .pydantic_validators import validate_torch_tensor
from .pydantic_validators import validate_np_array
from .pydantic_validators import validate_np_scalar

from abc import ABC


class NpScalarType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod
    def __get_validators__(cls):
        yield validate_np_scalar


class NpTorchType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_np_torch

    @classmethod
    def validate_np_torch(cls, field_type):
        try:
            return validate_torch_tensor(field_type)
        except ValueError:
            pass

        try:
            return validate_np_array(field_type)
        except ValueError:
            pass


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
        yield validate_nn_module


class LossType(ModuleType):
    pass


class FunctionType(ABC):
    @classmethod
    def __get_validators__(cls):
        yield validate_function


class OptimizerType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """

    @classmethod
    def __get_validators__(cls):
        yield validate_nn_optimizer


class SchedulerType(ABC):
    """
    A generic array/tensor type that returns true for every 
    f. call isinstance() with any torch.tensor and np.ndarray
    """

    @classmethod
    def __get_validators__(cls):
        yield validate_nn_scheduler
