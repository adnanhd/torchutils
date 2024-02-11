import typing
import torch
import inspect
from ...utils import BuilderType
from ._dev_utils import obtain_registered_kwargs

try:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler
except AttributeError:
    LRScheduler = torch.optim.lr_scheduler._LRScheduler


class Scheduler(BuilderType):    
    @classmethod
    def builder_from_class(cls, field_type, info):
        if inspect.isclass(field_type) and cls.__subclasscheck__(field_type):
            optimizer = info.data['optimizer']
            kwargs = obtain_registered_kwargs(field_type, info.data['arguments'])
            return field_type(optimizer, **kwargs)
        return field_type



Scheduler.register_subclasses(LRScheduler)
Scheduler.register(torch.optim.lr_scheduler.ReduceLROnPlateau)
