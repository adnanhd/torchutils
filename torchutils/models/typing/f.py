import typing
import torch
from torch.nn.modules.loss import _Loss as Loss
from .utils import _BaseValidator, reverse_dict, obtain_registered_kwargs
from .tensor import Tensor


class Functional(_BaseValidator):
    TYPE = typing.Callable[[Tensor.TYPE, Tensor.TYPE], Tensor.TYPE]
    __typedict__ = dict()

    @classmethod
    def class_validator(cls, field_type, info):
        if not isinstance(field_type, cls.TYPE):
            raise ValueError(f"{field_type} is not a {cls.TYPE}")
        return field_type


class Criterion(_BaseValidator):
    TYPE = typing.Union[torch.nn.Module, Functional]
    __typedict__ = dict()

    @classmethod
    def class_validator(cls, field_type, info):
        if isinstance(field_type, str):
            field_class = cls.__typedict__[field_type]
            kwargs = obtain_registered_kwargs(field_class, info.data['arguments'])
            field_type = field_class(**kwargs)
        if not (isinstance(field_type, torch.nn.Module) or callable(field_type)):
            raise ValueError(f"{field_type} is not a {cls.TYPE}")
        return field_type
    

for loss_func in vars(torch.functional.F).values():
    if callable(loss_func):
        Functional.__set_component__(loss_func)


for criterion_class in vars(torch.nn.modules.loss).values():
    if hasattr(criterion_class, 'mro') \
        and Loss in criterion_class.mro() \
        and criterion_class is not Loss:
        Criterion.__set_component__(criterion_class)
