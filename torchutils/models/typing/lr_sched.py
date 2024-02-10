import typing
import torch
import inspect
from .base import ClassNameRegistrar, _BaseModelType, obtain_registered_kwargs

try:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler
except AttributeError:
    LRScheduler = torch.optim.lr_scheduler._LRScheduler


class Scheduler(_BaseModelType, metaclass=ClassNameRegistrar):
    @classmethod
    def model_name_validator(cls, field_type, info):
        if inspect.isclass(field_type):
            if field_type not in cls.__subclasses_list__():
                raise ValueError(f"Unknown model {field_type}")
            optimizer = info.data['optimizer']
            kwargs = obtain_registered_kwargs(field_type, info.data['arguments'])
            return field_type(optimizer, **kwargs)
        return field_type


Scheduler.register_subclasses(LRScheduler)
Scheduler.register(torch.optim.lr_scheduler.ReduceLROnPlateau)
