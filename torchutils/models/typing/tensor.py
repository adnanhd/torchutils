import numpy
import torch
from .utils import _BaseModel


class Tensor(_BaseModel):
    """
    A generic array/tensor type that returns true for every
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod
    def tensor_validator(cls, field_type, info=None):
        if isinstance(field_type, torch.Tensor):
            return field_type
        raise ValueError(f"{field_type} is not a {torch.Tensor}")

    @classmethod
    def ndarray_validator(cls, field_type, info=None):
        if isinstance(field_type, numpy.ndarray):
            return field_type
        raise ValueError(f"{field_type} is not a {numpy.ndarray}")


Tensor.register(torch.Tensor)


class GradTensor(Tensor):
    """
    A generic array/tensor type that returns true for every
    f. call isinstance() with any torch.tensor and np.ndarray
    """
    @classmethod
    def __get_validators__(cls):
        yield cls.field_validator
        yield cls.grad_validator

    @classmethod
    def grad_validator(cls, field_type: torch.Tensor, info=None):
        if field_type.requires_grad:
            return field_type
        raise ValueError(f"{field_type} doesn't require gradient.")
