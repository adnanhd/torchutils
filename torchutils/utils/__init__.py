#!/usr/bin/env python

import numpy as np
import torch
from .version_v2 import Version
from .fnctutils import overload
from .mthdutils import hybridmethod
from .decorators import verbose, profile
from .config import INIObject
from .hash import Hashable
from .mappings import (
    string_to_criterion_class,
    string_to_optimizer_class,
    string_to_scheduler_class,
    string_to_functionals,
    string_to_types,
)


def reverse_dict(dictionary: dict):
    def reverse_item(item: tuple): return (item[1], item[0])
    return dict(map(reverse_item, dictionary.items()))


types_to_string = reverse_dict(string_to_types)
criterion_class_to_string = reverse_dict(string_to_criterion_class)
optimizer_class_to_string = reverse_dict(string_to_optimizer_class)
scheduler_class_to_string = reverse_dict(string_to_scheduler_class)
functionals_to_string = reverse_dict(string_to_functionals)


_str2types = {
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

_types2str = reverse_dict(_str2types)
