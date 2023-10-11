import numpy
import torch
import typing
from .utils import _BaseValidator


class Tensor(_BaseValidator):
    TYPE = typing.Union[torch.Tensor, numpy.ndarray]
    """
    A generic array/tensor type that returns true for every
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod
    def class_validator(cls, field_type):
        if isinstance(field_type, torch.Tensor) \
                or isinstance(field_type, numpy.ndarray):
            return field_type


class GradTensor(_BaseValidator):
    TYPE = torch.Tensor
    """
    A generic array/tensor type that returns true for every
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod
    def class_validator(cls, field_type):
        if isinstance(field_type, torch.Tensor) \
                and field_type.requires_grad:
            return field_type
