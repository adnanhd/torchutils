from abc import ABC
import typing

from .validators import validate_criterion
from .validators import validate_nn_scheduler
from .validators import validate_nn_optimizer
from .validators import validate_nn_module
from .validators import validate_torch_dataloader
from .validators import validate_torch_dataset
from .validators import validate_torch_tensor
from .validators import validate_np_array
from .validators import validate_np_scalar


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


class GradTensorType(ABC):
    """
    A generic array/tensor type that returns true for every
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_np_torch

    @classmethod
    def validate_np_torch(cls, field_type):
        field_type = validate_torch_tensor(field_type)
        if not field_type.requires_grad:
            raise ValueError(
                "Tensor must have a gradient"
            )


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


class FunctionalType(ABC):
    @classmethod
    def __get_validators__(cls):
        yield validate_criterion


CriterionType = typing.Union[ModuleType, FunctionalType]


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
