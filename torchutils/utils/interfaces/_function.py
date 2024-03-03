from ._base import _BaseModelType, RegisterationError
from abc import ABCMeta
import types

class FunctionalRegistrar(ABCMeta):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        cls._registrar = dict()  # Initialize the dictionary for instances
        return cls

    def register(cls, instance):
        """Registers an instance with its class name as the key."""
        assert isinstance(instance, FunctionalType), f'{instance} is not of type FunctionalType'
        if instance.__name__ in cls._registrar:
            raise RegisterationError(f'class {instance.__name__}: already registered')
        cls._registrar[instance.__name__] = instance
        return instance

    def get_functional(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        try:
            return cls._registrar[class_name]
        except KeyError:
            raise RegisterationError(f"class {class_name}: not registered")    

    def has_functional(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        return class_name in cls._registrar.keys()
    

class FunctionalType(_BaseModelType, metaclass=FunctionalRegistrar):
    @classmethod
    def register(cls, field_type):
        field_type = cls.field_validator(field_type=field_type, info=None)
        return FunctionalRegistrar.register(cls, field_type)

    @classmethod
    def __get_validators__(cls):
        yield cls.get_function_from_name
        yield cls.field_validator
        yield cls.field_signature_validator

    @classmethod
    def get_function_from_name(cls, field_type, info):
        if isinstance(field_type, str) and cls.has_functional(field_type):
            return cls.get_functional(field_type)
        return field_type

    @classmethod
    def field_validator(cls, field_type, info):
        assert isinstance(field_type, FunctionalType), f'{field_type} is failed'
        return field_type

    @classmethod
    def field_signature_validator(cls, field_type, info):
        return field_type
    


## registary
ABCMeta.register(FunctionalType, types.FunctionType)
