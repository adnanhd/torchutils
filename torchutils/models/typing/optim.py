import torch.optim as optim
from .utils import _BaseValidator, obtain_registered_kwargs


class Optimizer(_BaseValidator):
    TYPE = optim.Optimizer
    __typedict__ = dict()

    @classmethod
    def class_validator(cls, field_type, info):
        if isinstance(field_type, str):
            field_class = cls.__typedict__[field_type]
            params = info.data['model'].parameters()
            kwargs = obtain_registered_kwargs(field_class, info.data['arguments'])
            field_type = field_class(params, **kwargs)
        if not isinstance(field_type, cls.TYPE):
            raise ValueError(f"{field_type} is not a {cls.TYPE}")
        return field_type


for optimizer_class in optim.Optimizer.__subclasses__():
    Optimizer.__set_component__(optimizer_class)
