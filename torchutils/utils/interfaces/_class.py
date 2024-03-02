from ._base import _BaseModelType, RegisterationError
import inspect
import abc


class ClassNameRegistrar(abc.ABCMeta):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        cls._registrar = dict()  # Initialize the dictionary for instances
        return cls

    def register(cls, instance):
        """Registers an instance with its class name as the key."""
        assert inspect.isclass(instance)
        if instance.__name__ in cls._registrar:
            raise RegisterationError(f'class {instance.__name__}: already registered')
        cls._registrar[instance.__name__] = instance
        return super().register(instance)

    def register_subclasses(cls, instance):
        """Registers an instance with its class name as the key."""
        tuple(map(cls.register, instance.__subclasses__()))
        return instance

    def get_subclass(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        try:
            return cls._registrar[class_name]
        except KeyError:
            raise RegisterationError(f"class {class_name}: not registered")

    def has_subclass(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        return class_name in cls._registrar.keys()
    


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