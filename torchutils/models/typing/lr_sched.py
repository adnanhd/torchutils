import typing
import torch
from .utils import _BaseValidator, reverse_dict, obtain_registered_kwargs


class Scheduler(_BaseValidator):
    TYPE = typing.Union[torch.optim.lr_scheduler.LRScheduler,
                        torch.optim.lr_scheduler.ReduceLROnPlateau]
    __typedict__ = dict()

    @classmethod
    def class_validator(cls, field_type, info):
        if isinstance(field_type, str):
            field_class = cls.__typedict__[field_type]
            optimizer = info.data['optimizer']
            kwargs = obtain_registered_kwargs(field_class, info.data['arguments'])
            field_type = field_class(optimizer, **kwargs)
        if not isinstance(field_type, cls.TYPE):
            raise ValueError(f"{field_type} is not a {cls.TYPE}")
        return field_type
    

for scheduler in torch.optim.lr_scheduler.LRScheduler.__subclasses__():
    Scheduler.__set_component__(scheduler)
Scheduler.__set_component__(torch.optim.lr_scheduler.ReduceLROnPlateau)