from ._base import RegisterationError
from pydantic import BaseModel
from abc import ABCMeta


__all__ = ['InstanceRegistrar', 'RegisteredBaseModel']


class InstanceRegistrar(ABCMeta):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        cls._instances = dict()  # Initialize the dictionary for instances
        return cls

    def add_instance(cls, name: str, obj):
        cls._instances[name] = obj

    def get_instance(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        try:
            return cls._instances[class_name]
        except KeyError:
            raise RegisterationError(f"class {class_name}: not registered") 

    def has_instance(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        return class_name in cls._instances.keys()


class RegisteredBaseModel(BaseModel):
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


_REGISTRY = {}