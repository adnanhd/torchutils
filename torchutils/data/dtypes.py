import torch
import numpy as np
from abc import ABC, ABCMeta
from typing import Union, Optional
from pydantic import BaseModel, validator
from torchutils.utils import string_to_types, Hashable, hybridmethod
from torchutils.utils.pydantic import NpScalarType, NpTorchType


class DType(BaseModel, Hashable):
    dtype: Union[type, str, NpScalarType, NpTorchType]
    shape: Union[tuple, torch.Size, None] = None

    def __init__(self, dtype, shape=None) -> None:
        super(DType, self).__init__(dtype=dtype, shape=shape)
    
    @validator('dtype')
    @classmethod
    def string_to_dtype(cls, dtype):
        if not isinstance(dtype, str):
            return dtype
        dtype = dtype.lower().strip()
        assert dtype in string_to_types
        return string_to_types[dtype]

    #@validator('shape')
    #@classmethod
    #def check_shape(cls, shape):
    #    if NpTorchType.validate(cls.dtype):
    #        return tuple(shape)

    def __instancecheck__(self, other):
        return isinstance(other, (np.ndarray, torch.Tensor)) and other.dtype == self.dtype or isinstance(other, self.dtype)
    
    def __hash__(self):
        return int(self._md5(), 16)

    def __call__(self, value):
        value = self.dtype(value)
        if self.shape is None: return value
        else: return value.reshape(self.shape)

    def __repr__(self):
        if self.shape is None: return f'{self.__class__.__name__}(dtype={self.dtype})'
        else: return f'{self.__class__.__name__}(dtype={self.dtype}, shape={self.shape})'

