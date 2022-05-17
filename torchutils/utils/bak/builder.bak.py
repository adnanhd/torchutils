import os
import math
import copy
import torch
import numpy as np

from .dtypes import Datum
from .utils import hybridmethod

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

_str2types = {
    'string': str,
    'integer': int,
    'str': str,
    'int': int,
    'float': float,
    'bool': bool,
}

_types2str = {
    t: s for s, t in _str2types.items()
}


class Builder:
    """ A dataset preprocessor. """
    __slots__ = ('_values', '_device')

    @dataclass
    class Value:
        _dtype: Union[type, str, torch.dtype, np.dtype] = field()
        _shape: Optional[Union[  torch.Size,  tuple]] = field(default=None)

        def __init__(self, dtype, shape=None):
            self.dtype = dtype
            self.shape = shape

        @property
        def dtype(self):
            if not isinstance(self._dtype, str):
                return self._dtype
            else:
                assert self._dtype.lower() in _str_types
                return _str_types[self._dtype.lower()]

        @dtype.setter
        def dtype(self, dtype) -> None:
            allowed_dtypes = self.__annotations__['_dtype']._subs_tree()[1:]
            assert any(isinstance(dtype, dt) for dt in allowed_dtypes)
            self._dtype = dtype

        @property
        def shape(self):
            return self._shape

        @shape.setter
        def shape(self, shape) -> None:
            allowed_shapes = self.__annotations__['_shape']._subs_tree()[1:]
            assert any(isinstance(shape, sh) for sh in allowed_shapes)
            self._shape = shape

        def check_shape(self, other: "Value") -> bool:
            assert isinstance(other, self.__class__)
            return self.shape == other.shape

        def check_dtype(self, other: "Value") -> bool:
            assert isinstance(other, self.__class__)
            return self.dtype == other.dtype

        def __eq__(self, other: "Value") -> bool:
            assert isinstance(other, self.__class__)
            return self.check_shape(other) and self.check_dtype(other)

        def __repr__(self):
            dtype = _types2str[self._dtype] if isinstance(self._dtype, type) else self._dtype
            return f'{self.__class__.__name__}(dtype={dtype}, shape={self._shape})'
    
    def __init__(self, *args, device=torch.device('cpu'), dtype=torch.float64, **values):
        assert isinstance(device, str) or isinstance(device, torch.device)
        assert isinstance(dtype, type) or isinstance(dtype, torch.dtype)
        self._values = dict()
        def make_value(value):
            if not isinstance(value, self.Value):
                if isinstance(value, dict):
                    value = self.Value(**value)
                elif isinstance(value, list) or isinstance(value, tuple):
                    value = self.Value(*value)
                else:
                    value = self.Value(value)
            return value
        
        for name, value in enumerate(args):
            self._values[name] = make_value(value)

        for name, value in values.items():
            self._values[name] = make_value(value)

    def __len__(self):
        return len(self._values)

    def __getitem__(self, index):
        return self._values[index]
       
    def __getattr__(self, key):
        if key in self._values:
            return self._values[key]
        else:
            return self.__slots__.index(key)

    def set_shape_of(self, index, shape):
        self.__getattr__(index).shape = shape

    def get_shape_of(self, index):
        return self.__getattr__(index).shape

    def set_dtype_of(self, index, dtype):
        self.__getattr__(index).dtype = dtype

    def get_dtype_of(self, index):
        return self.__getattr__(index).dtype

    def generate_example(self, path):
        pass

