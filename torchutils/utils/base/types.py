import abc
import typing
from .extra import ClassNameRegistrar


class _BaseModelType(abc.ABC):
    @classmethod
    def __get_validators__(cls):
        yield cls.field_validator

    @classmethod
    def field_validator(cls, field_type, info):
        if isinstance(field_type, cls):
            return field_type
        raise ValueError(f"{field_type} is not a {cls}")


class _BaseModelTypeWithRegistrar(_BaseModelType, metaclass=ClassNameRegistrar):
    @classmethod
    def __get_validators__(cls):
        yield cls.class_name_validator
        yield from super().__get_validators__

    @classmethod
    def class_name_validator(cls, field_type, info):
        if isinstance(field_type, str):
            return cls.get_subclass(field_type)
        return field_type