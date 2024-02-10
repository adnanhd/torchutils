import abc
import inspect
import pydantic
from .extra import ClassNameRegistrar, FunctionalRegistrar, InstanceRegistrar


class _BaseModelType(abc.ABC):
    @classmethod
    def __get_validators__(cls):
        yield cls.field_validator

    @classmethod
    def field_validator(cls, field_type, info):
        if isinstance(field_type, cls):
            return field_type
        raise ValueError(f"{field_type} is not a {cls}")


class BuilderType(_BaseModelType, metaclass=ClassNameRegistrar):
    @classmethod
    def __get_validators__(cls):
        yield cls.get_class_from_name
        yield cls.builder_from_class
        yield cls.field_validator

    @classmethod
    def get_class_from_name(cls, field_type, info):
        if isinstance(field_type, str) and cls.has_subclass(field_type):
            return cls.get_subclass(field_type)
        return field_type
    
    @abc.abstractclassmethod
    def builder_from_class(cls, field_type, info):
        pass


class FunctionalType(_BaseModelType, metaclass=FunctionalRegistrar):
    @classmethod
    def register(cls, field_type):
        field_type = cls.field_validator(field_type=field_type, info=None)
        return FunctionalRegistrar.register(cls, field_type)

    @classmethod
    def __get_validators__(cls):
        yield cls.field_validator
        yield cls.field_signature_validator

    @classmethod
    def field_validator(cls, field_type, info):
        assert inspect.isfunction(field_type)
        return field_type
    
    @classmethod
    def field_signature_validator(cls, field_type, info):
        return field_type


_REGISTRY = {}

class RegisteredBaseModel(pydantic.BaseModel):
    name: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _REGISTRY[cls.__name__] = dict()

    def __init__(self, name: str, **kwargs):
        super().__init__(name=name, **kwargs)
        _REGISTRY[self.__class__.__name__][name] = self
    
    def __hash__(self) -> int:
        return self.name.__hash__()
    
    @classmethod
    def list_instance_names(cls):
        return _REGISTRY[cls.__name__].keys()
    
    @classmethod
    def list_instances(cls):
        return _REGISTRY[cls.__name__].values()

    @classmethod
    def get_instance(cls, name: str):
        """Retrieves the registered instance by its class name."""
        try:
            return _REGISTRY[cls.__name__][name]
        except KeyError:
            raise ValueError(f"{cls.__name__}: {name} not registered") from None      

    @classmethod
    def has_instance(cls, name: str):
        """Retrieves the registered instance by its class name."""
        return name in _REGISTRY[cls.__name__].keys()
