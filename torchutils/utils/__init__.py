#!/usr/bin/env python
import torchutils.utils.pydantic
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
    _str2types, _types2str
)


def reverse_dict(dictionary: dict):
    def reverse_item(item: tuple): return (item[1], item[0])
    return dict(map(reverse_item, dictionary.items()))


types_to_string = reverse_dict(string_to_types)
criterion_class_to_string = reverse_dict(string_to_criterion_class)
optimizer_class_to_string = reverse_dict(string_to_optimizer_class)
scheduler_class_to_string = reverse_dict(string_to_scheduler_class)
functionals_to_string = reverse_dict(string_to_functionals)
