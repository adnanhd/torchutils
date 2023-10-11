import typing
import torch
import inspect
from torch.nn.modules.loss import _Loss as Loss
from .utils import _BaseValidator, obtain_registered_kwargs
from .tensor import Tensor


class Functional(_BaseValidator):
    TYPE = typing.Callable[[Tensor.TYPE, Tensor.TYPE], Tensor.TYPE]
    __typedict__ = dict()

    @classmethod
    def class_validator(cls, field_type, info):
        if inspect.isfunction(field_type) and \
                keys.issubset(set(inspect.signature(field_type).parameters.keys())):
            return field_type
        raise ValueError(f"{field_type} is not a {cls.TYPE}")


class Criterion(_BaseValidator):
    TYPE = typing.Union[torch.nn.Module, Functional]
    __typedict__ = dict()

    @classmethod
    def class_validator(cls, field_type, info):
        try:
            return Functional.class_validator(field_type, info)
        except ValueError:
            if isinstance(field_type, str):
                field_class = cls.__typedict__[field_type]
                if inspect.isfunction(field_class):
                    return field_class
                elif inspect.isclass(field_class):
                    kwargs = info.data['arguments']
                    kwargs = obtain_registered_kwargs(field_class, kwargs)
                    field_type = field_class(**kwargs)
                else:
                    raise ValueError(f"{field_type} is not a {cls.TYPE}")
            if isinstance(field_type, torch.nn.Module):
                return field_type
            raise ValueError(f"{field_type} is not a {cls.TYPE}")


keys = {'input', 'target'}
for loss_func in vars(torch.functional.F).values():
    if inspect.isfunction(loss_func) and \
            keys.issubset(set(inspect.signature(loss_func).parameters.keys())):
        Functional.__set_component__(loss_func)
        Criterion.__set_component__(loss_func)


for criterion_class in vars(torch.nn.modules.loss).values():
    if hasattr(criterion_class, 'mro') \
            and Loss in criterion_class.mro() \
            and criterion_class is not Loss:
        Criterion.__set_component__(criterion_class)
