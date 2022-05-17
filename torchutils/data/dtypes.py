import os
import re
import torch
import hashlib
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from .utils import _str2types, _types2str

from copy import deepcopy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, Optional
from pydantic import BaseModel, validator
from .utils import Hashable


class TorchTensorType(type):  
    @classmethod  
    def __get_validators__(cls):  
        yield cls.validate

    @classmethod
    def validate(cls, dtype: torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        else:
            raise ValueError('invalid torch.dtype')


class Value(BaseModel, Hashable):
    dtype: Union[type, str, TorchTensorType]
    shape: Optional[Union[tuple, torch.Size]] = None

    def __init__(self, dtype, shape=None) -> None:
        super().__init__(dtype=dtype, shape=shape)

    @validator('dtype')
    @classmethod
    def string_to_dtype(cls, dtype):
        if not isinstance(dtype, str):
            return dtype
        dtype = dtype.lower().strip()
        assert dtype in _str2types
        return _str2types[dtype]
    
    def __hash__(self):
        return int(self._md5(), 16)

    def __repr__(self):
        dtype = _types2str[self.dtype] if isinstance(self.dtype, type) else self.dtype
        return f'{self.__class__.__name__}(dtype={dtype}, shape={self.shape})'

    def __str__(self):
        dtype = _types2str[self.dtype] if isinstance(self.dtype, type) else self.dtype
        return f'(dtype={dtype}, shape={self.shape})'


class TensorValue(Value):
    pass

