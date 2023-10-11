from abc import ABC
import warnings


class RegisterationWarning(Warning):
    pass


warnings.filterwarnings('ignore', category=RegisterationWarning)


def reverse_dict(d: dict) -> dict:
    assert isinstance(d, dict)
    if d.__len__() == 0:
        return dict()
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
            warnings.warn(f"{cls.__qualname__}: {name} not registered", RegisterationWarning)
        return cls.__typedict__[name]

    @classmethod
    def __set_component__(cls, type):
        name = type.__name__
        if name in cls.__typedict__:
            warnings.warn(f"{cls.__qualname__}: {name} yet registered", RegisterationWarning)
        cls.__typedict__[name] = type

    @classmethod
    def isinstance(cls, obj):
        try:
            return all(not validator(obj) is None
                       for validator in cls.__get_validators__())
        except ValueError:
            return False

    @classmethod
    def class_validator(cls, field_type, info):
        raise NotImplementedError()
