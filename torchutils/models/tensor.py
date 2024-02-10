import numpy
import torch
from ..utils import _BaseModelType


class Tensor(_BaseModelType):
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
    def field_validator(cls, field_type: torch.Tensor, info=None):
        field_type = super().field_validator(field_type=field_type, info=info)
        if field_type.requires_grad:
            return field_type
        raise ValueError(f"{field_type} doesn't require gradient.")
