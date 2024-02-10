from ...utils import _BaseModelType, ClassNameRegistrar, FunctionalRegistrar
import inspect
import abc


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

    @classmethod
    def field_validator(cls, field_type, info):
        assert inspect.isfunction(field_type)
        cls.signature_assertions(inspect.signature(field_type))
        return field_type

    @abc.abstractclassmethod
    def signature_assertions(cls, signature: inspect.Signature):
        pass