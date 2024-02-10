
import typing
import abc
import inspect


class RegisterationError(ValueError):
    pass


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
            raise RegisterationError(f"class {class_name}: not registered") from None

    def has_subclass(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        return class_name in cls._registrar.keys()
    

class FunctionalRegistrar(abc.ABCMeta):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        cls._registrar = dict()  # Initialize the dictionary for instances
        return cls

    def register(cls, instance):
        """Registers an instance with its class name as the key."""
        assert inspect.isfunction(instance)
        if instance.__name__ in cls._registrar:
            raise RegisterationError(f'class {instance.__name__}: already registered')
        cls._registrar[instance.__name__] = instance
        return instance

    def get_functional(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        try:
            return cls._registrar[class_name]
        except KeyError:
            raise RegisterationError(f"class {class_name}: not registered") from None      

    def has_functional(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        return class_name in cls._registrar.keys()


class InstanceRegistrar(abc.ABCMeta):
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
            raise RegisterationError(f"class {class_name}: not registered") from None      

    def has_instance(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        return class_name in cls._instances.keys()
