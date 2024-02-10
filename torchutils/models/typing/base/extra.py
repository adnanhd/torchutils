
import typing
import abc


class RegisterationError(ValueError):
    pass


class ClassNameRegistrar(abc.ABCMeta):

    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        cls._instances = {}  # Initialize the dictionary for instances
        return cls

    def register(cls, instance):
        """Registers an instance with its class name as the key."""
        if instance.__name__ in cls._instances:
            raise RegisterationError(f'class {instance.__name__}: already registered')
        cls._instances[instance.__name__] = instance
        return super().register(instance)

    def register_subclasses(cls, instance):
        """Registers an instance with its class name as the key."""
        tuple(map(cls.register, instance.__subclasses__()))
        return instance
            

    def get_instance(cls, class_name: str):
        """Retrieves the registered instance by its class name."""
        try:
            return cls._instances[class_name]
        except KeyError:
            raise RegisterationError(f"class {class_name}: not registered") from None
