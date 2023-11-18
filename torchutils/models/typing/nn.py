import torch.nn as nn
import torchvision.models as m
from .utils import _BaseModel, obtain_registered_kwargs

__all__ = ['NeuralNet']


class NeuralNet(_BaseModel):

    @classmethod
    def __get_validators__(cls):
        yield cls.torchvision_model_builder
        yield cls.field_validator

    @classmethod
    def torchvision_model_builder(cls, field_type, info):
        if isinstance(field_type, str):
            builder = m.get_model_builder(field_type)
            config = obtain_registered_kwargs(builder, info.data['arguments'])
            return builder(**config)
        return field_type


NeuralNet.register(nn.Module)
