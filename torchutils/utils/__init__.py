#!/usr/bin/env python

from .version import Version
from .fnctutils import overload
from .mthdutils import hybridmethod
from .decorators import verbose, profile
from .config import INIObject
from .hash import Hashable
import torchutils.utils.config
#import torchutils.data.utils.preprocessing
import numpy as np
import torch

string_to_types = {
    'string': str, 'str': str,
    'integer': int, 'int': int,
    'boolean': bool, 'bool': bool,
    'float': float,
    'torch.float16': torch.float16, 
    'torch.float32': torch.float32,
    'torch.float64': torch.float64,
    'torch.int8': torch.int8,
    'torch.uint8': torch.uint8,
    'torch.int16': torch.int16,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'np.float16': np.float16, 
    'np.float32': np.float32, 
    'np.float64': np.float64,
    'np.float128': np.float128,
    'np.int8':  np.int8,
    'np.int16': np.int16,
    'np.int32': np.int32,
    'np.int64': np.int64,
    'np.uint8':  np.uint8,
    'np.uint16': np.uint16,
    'np.uint32': np.uint32,
    'np.uint64': np.uint64,
}

types_to_string = {t: s for s, t in string_to_types.items()}

_str2types = string_to_types
_types2str = types_to_string
