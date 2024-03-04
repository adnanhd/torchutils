from typing import Any
from pydantic_core import CoreSchema, core_schema
from pydantic import GetCoreSchemaHandler, TypeAdapter
from abc import ABCMeta

class RegisterationError(Exception):
    pass

class Registrar(ABCMeta):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        cls._registrar = dict()  # Initialize the dictionary for instances
        return cls

    def register(cls, instance):
        """Registers an instance with its class name as the key."""
        #assert isinstance(instance, FunctionalType), f'{instance} is not of type FunctionalType'
        if instance.__name__ in cls._registrar:
            raise RegisterationError(f'class {instance.__name__}: already registered')
        cls._registrar[instance.__name__] = instance
        return instance

    def get_from_name(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        try:
            return cls._registrar[class_name]
        except KeyError:
            raise RegisterationError(f"class {class_name}: not registered")    

    def is_registered(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        return class_name in cls._registrar.keys()
    

class Username(str, metaclass=Registrar):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))
