import abc
import typing
from extra import ClassNameRegistrar


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
            return cls.get_instance(field_type)
        return field_type


class _RegisteredBasModelv2(_BaseModelType):
    @classmethod
    def __subclasses_dict__(cls) -> typing.Dict[str, type]:
        return dict(map(lambda subcls: (subcls.__name__, subcls),
                        cls.__subclasses_list__()))

    @classmethod
    def __get_validators__(cls):
        yield cls.class_name_validator
        yield cls.model_name_validator
        yield cls.field_validator

    @classmethod
    @abc.abstractmethod
    def __subclass_list__(cls) -> typing.List[type]:
        pass

    @classmethod
    def class_name_validator(cls, field_type, info):
        if isinstance(field_type, str):
            try:
                return cls.__subclasses_dict__()[field_type]
            except KeyError:
                raise ValueError(f"Unknown model {field_type}")
        return field_type

    @classmethod
    @abc.abstractmethod
    def model_name_validator(cls, field_type, info):
        pass
