#!/usr/bin/env python
from .mappings import (
    string_to_criterion_class,
    string_to_optimizer_class,
    string_to_scheduler_class,
    string_to_functionals,
)

from .debug import PrintOnce
from .hash import Hashable, digest_numpy, digest_torch, digest
from .config import BaseConfig
from .decorators import verbose, profile
from .mthdutils import hybridmethod
from .fnctutils import overload, obtain_registered_kwargs
from .version_v2 import Version
from .inspect_ import isiterable, islistlike, issubscriptable
import torch
import numpy as np


def reverse_dict(dictionary: dict):
    def reverse_item(item: tuple): return (item[1], item[0])
    return dict(map(reverse_item, dictionary.items()))


criterion_class_to_string = reverse_dict(string_to_criterion_class)
optimizer_class_to_string = reverse_dict(string_to_optimizer_class)
scheduler_class_to_string = reverse_dict(string_to_scheduler_class)
functionals_to_string = reverse_dict(string_to_functionals)
