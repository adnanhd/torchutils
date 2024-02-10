import torch.nn as nn
import torchvision.models as m
import inspect
import abc
from ...utils import BuilderType
from ._dev_utils import obtain_registered_kwargs

__all__ = ['NeuralNet']


class NeuralNet(BuilderType):
    @classmethod
    def __get_validators__(cls):
        yield cls.get_class_from_name
        yield cls.torchvision_builder_from_name
        yield cls.builder_from_class
        yield cls.field_validator
            
    @classmethod
    def builder_from_class(cls, field_type, info):
        if inspect.isclass(field_type) and cls.__subclasscheck__(field_type) \
            or inspect.isfunction(field_type):
            config = obtain_registered_kwargs(field_type, info.data['arguments'])
            device = info.data['arguments'].get('device', None)
            return field_type(**config).to(device=device)
        return field_type

    @classmethod
    def torchvision_builder_from_name(cls, field_type, info):
        if isinstance(field_type, str) and field_type in m.list_models():
            return m.get_model_builder(field_type)
        return field_type


abc.ABCMeta.register(NeuralNet, nn.Module)
