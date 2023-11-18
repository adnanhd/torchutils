import typing
import torch
import inspect
from .utils import _RegisteredBasModelv2, obtain_registered_kwargs


FUNCTIONALS = list()


def register_functionals(fn):
    FUNCTIONALS.append(fn)


class Functional(_RegisteredBasModelv2):
    @classmethod
    def __subclasses_list__(cls) -> typing.List[type]:
        return FUNCTIONALS.copy()

    @classmethod
    def __get_validators__(cls):
        yield cls.class_name_validator
        yield cls.function_validator
        yield cls.loss_signature_validator

    @classmethod
    def function_validator(cls, field_type, info):
        if inspect.isfunction(field_type):
            return field_type
        raise ValueError(f"{field_type} is not a {cls.__name__}")

    @classmethod
    def loss_signature_validator(cls, field_type, info):
        sign = inspect.signature(field_type).parameters.keys()
        if {'input', 'target'}.issubset(set(sign)):
            return field_type
        raise ValueError(f"{field_type} is not a {cls.__name__}")


for loss_func in vars(torch.functional.F).values():
    try:
        loss_func = Functional.function_validator(loss_func, None)
        loss_func = Functional.loss_signature_validator(loss_func, None)
        FUNCTIONALS.append(loss_func)
    except ValueError:
        pass


class Criterion(_RegisteredBasModelv2):
    @classmethod
    def __subclasses_list__(cls) -> typing.List[type]:
        return torch.nn.modules.loss._Loss.__subclasses__()

    @classmethod
    def model_name_validator(cls, field_type, info):
        if inspect.isclass(field_type):
            if field_type not in cls.__subclasses_list__():
                raise ValueError(f"Unknown model {field_type}")
            kwargs = obtain_registered_kwargs(field_type, info.data['arguments'])
            return field_type(**kwargs)
        return field_type


Criterion.register(torch.nn.modules.loss._Loss)
