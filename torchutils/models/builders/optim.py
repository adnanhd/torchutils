import torch.optim as optim
import typing
import inspect
from ...utils import BuilderType
from ._dev_utils import obtain_registered_kwargs


class Optimizer(BuilderType):
    @classmethod
    def builder_from_class(cls, field_type, info):
        if inspect.isclass(field_type) and cls.__subclasscheck__(field_type):
            model = info.data['model']
            kwargs = obtain_registered_kwargs(field_type, info.data['arguments'])
            return field_type(model.parameters(), **kwargs)
        return field_type


Optimizer.register_subclasses(optim.Optimizer)
