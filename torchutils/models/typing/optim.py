import torch.optim as optim
import typing
import inspect
from .utils import obtain_registered_kwargs, _RegisteredBasModelv2


class Optimizer(_RegisteredBasModelv2):
    @classmethod
    def __subclasses_list__(cls) -> typing.List[type]:
        return optim.Optimizer.__subclasses__()

    @classmethod
    def model_name_validator(cls, field_type, info):
        if inspect.isclass(field_type):
            if field_type not in cls.__subclasses_list__():
                raise ValueError(f"Unknown model {field_type}")
            model = info.data['model']
            kwargs = obtain_registered_kwargs(field_type, info.data['arguments'])
            return field_type(model.parameters(), **kwargs)
        return field_type


Optimizer.register(optim.Optimizer)
