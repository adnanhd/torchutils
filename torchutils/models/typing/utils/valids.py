from abc import ABC
import warnings


def reverse_dict(d: dict) -> dict:
    assert isinstance(d, dict)
    if d.__len__() == 0: return dict()
    return dict(map(lambda k, v: (v, k), *zip(*d.items())))


class _BaseValidator(ABC):
    TYPE = None
    __typedict__ = None

    @classmethod
    def __get_validators__(cls):
        yield cls.class_validator
    
    @classmethod
    def __get_component__(cls, name):
        if name not in cls.__typedict__:
            warnings.warn(f"{cls.__qualname__}: {name} not registered", Warning)
            #raise KeyError()
        return cls.__typedict__[name]
    
    @classmethod
    def __set_component__(cls, type):
        name = type.__name__
        if name in cls.__typedict__:
            warnings.warn(f"{cls.__qualname__}: {name} yet registered", Warning)
            #raise KeyError(f"{cls.__qualname__}: {name} yet registered")
        cls.__typedict__[name] = type

    @classmethod
    def isinstance(cls, obj):
        return isinstance(obj, cls.TYPE)

    @classmethod
    def class_validator(cls, field_type, info):
        raise NotImplementedError()

"""
from typing import Optional, Any, Callable

def validate_nn_module(other: Any) -> Optional[Module]:
    if isinstance(other, Module):
        return other
    else:
        raise ValueError()


def validate_criterion(other: Any) -> Callable[[Any, Any], Any]:
    if callable(other):
        return other
    elif isinstance(other, str):
        if other in string_to_criterion_class:
            return string_to_criterion_class[other]
        else:
            raise ValueError(f"{other} not registered criterion")
    else:
        raise ValueError()


def validate_nn_optimizer(other) -> Optional[Optimizer]:
    if inspect.isclass(other) and Optimizer.__subclasscheck__(other):
        return other
    elif (not inspect.isclass(other)) and \
            Optimizer.__subclasscheck__(other.__class__):
        return other
    elif not isinstance(other, str):
        raise ValueError()
    elif other in optimizer_class_to_string.values():
        return string_to_optimizer_class[other]
    else:
        raise ValueError(f"{other} not registered Optimizer")


def validate_nn_scheduler(other) -> Optional[_LRScheduler]:
    if inspect.isclass(other) and \
            (_LRScheduler.__subclasscheck__(other) or
             other is ReduceLROnPlateau):
        return other
    elif (not inspect.isclass(other)) and \
            (_LRScheduler.__subclasscheck__(other.__class__) or
             isinstance(other, ReduceLROnPlateau)):
        return other
    elif not isinstance(other, str):
        raise ValueError()
    elif other in scheduler_class_to_string.values():
        return string_to_scheduler_class[other]
    else:
        raise ValueError(f"{other} not registered Scheduler")
"""